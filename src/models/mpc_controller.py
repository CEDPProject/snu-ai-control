import numpy as np
import cvxpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from dataclasses import dataclass

class MPCController:
    """MPC 기반 HVAC 제어 모델"""
    
    def __init__(self, params):
        """
        Args:
            params: 모델 설정 (cfg.MODELSETTING)
        """
        self.params = params
        self.metadata = params['METADATA']
        self.features = params['METADATA']['features']
        
        # 동역학 학습 방법 선택
        self.dynamics_method = params.get('dynamics_method', 'linear')  # 'linear' or 'koopman'
        self.koopman_order = params.get('koopman_order', 2)  # Koopman용 다항식 차수
        
        # MPC 설정
        self.prediction_horizon = params.get('prediction_horizon', 12)
        self.control_horizon = params.get('control_horizon', 6)
        
        # 제약조건 (스케일링된 값)
        self.constraints = {
            'co2_limit': params.get('co2_limit', 2.0),
            'pm10_limit': params.get('pm10_limit', 2.0),
            'temp_min': params.get('temp_min', -1.5),
            'temp_max': params.get('temp_max', 1.5)
        }
        
        # 가중치
        self.weights = {
            'power': params.get('power_weight', 1.0),
            'comfort': params.get('comfort_weight', 0.5),
            'control_change': params.get('control_change_weight', 0.1)
        }
        
        # Feature 인덱스 설정
        self._setup_indices()
        
        # 상태공간 모델 행렬 (표준 표기법)
        # x(k+1) = A*x(k) + B*u(k) + E*d(k)
        # y(k) = C*x(k) + D*u(k)
        self.A = None  # 상태 전이 행렬
        self.B = None  # 제어 입력 행렬
        self.E = None  # 외란 입력 행렬
        self.C = None  # 출력 행렬
        self.D = None  # 직접 전달 행렬
        
        # 측정 가능한 상태 설정
        self.observable_states = params.get('observable_states', 'all')
        
        # Koopman 관련
        self.koopman_transform = None
        self.A_lifted = None
        self.B_lifted = None
        self.E_lifted = None  # 외란 행렬 (Koopman)
        self.lifting_dim = None
        
        # 평가 메트릭
        self.evaluation_results = {}
        
    def _setup_indices(self):
        """Feature 인덱스 매핑"""
        self.idx_map = {name: i for i, name in enumerate(self.features)}
        
        # 상태 변수 인덱스
        self.state_idx = [
            self.idx_map.get('실내미세먼지(㎍/㎥)', -1),
            self.idx_map.get('실내초미세먼지(㎍/㎥)', -1),
            self.idx_map.get('실내이산화탄소(ppm)', -1),
            self.idx_map.get('실내온도(℃)', -1),
            self.idx_map.get('실내습도(%)', -1)
        ]
        self.state_idx = [i for i in self.state_idx if i >= 0]
        
        # 제어 변수 인덱스
        self.control_idx = [
            self.idx_map.get('환기풍량_ord', -1),
            self.idx_map.get('환기전원_ON', -1),
            self.idx_map.get('냉난방전원_ON', -1),
            self.idx_map.get('냉난방설정온도(℃)', -1)
        ]
        self.control_idx = [i for i in self.control_idx if i >= 0]
        
        # 외란 변수 인덱스
        self.disturbance_idx = [
            self.idx_map.get('실외미세먼지(㎍/㎥)', -1),
            self.idx_map.get('실외초미세먼지(㎍/㎥)', -1),
            self.idx_map.get('실외온도(℃)', -1),
            self.idx_map.get('실외습도(%)', -1)
        ]
        self.disturbance_idx = [i for i in self.disturbance_idx if i >= 0]
        
        # 전력 변수 인덱스
        self.power_idx = [
            self.idx_map.get('환기유효전력(W)', -1),
            self.idx_map.get('냉난방유효전력(W)', -1)
        ]
        self.power_idx = [i for i in self.power_idx if i >= 0]
    
    def learn_dynamics(self, train_data):
        """시스템 동역학 학습 (방법 선택)"""
        print(f"\n시스템 동역학 학습 시작 (방법: {self.dynamics_method})...")
        
        if self.dynamics_method == 'linear':
            self._learn_dynamics_linear(train_data)
        elif self.dynamics_method == 'koopman':
            self._learn_dynamics_koopman(train_data)
        else:
            raise ValueError(f"Unknown dynamics method: {self.dynamics_method}")
    
    def _learn_dynamics_linear(self, train_data):
        """선형 회귀를 사용한 시스템 동역학 학습"""
        
        # 데이터 추출
        X_state = train_data[:-1, self.state_idx]
        X_control = train_data[:-1, self.control_idx]
        X_dist = train_data[:-1, self.disturbance_idx]
        Y_next = train_data[1:, self.state_idx]
        
        n_states = len(self.state_idx)
        n_controls = len(self.control_idx)
        n_dist = len(self.disturbance_idx)
        
        # 상태 방정식 행렬 학습: x(k+1) = A*x(k) + B*u(k) + E*d(k)
        self.A = np.zeros((n_states, n_states))
        self.B = np.zeros((n_states, n_controls))
        self.E = np.zeros((n_states, n_dist))
        
        for i in range(n_states):
            X = np.hstack([X_state, X_control, X_dist])
            y = Y_next[:, i]
            
            model = LinearRegression() 
            model.fit(X, y)
            
            self.A[i, :] = model.coef_[:n_states]
            self.B[i, :] = model.coef_[n_states:n_states+n_controls]
            self.E[i, :] = model.coef_[n_states+n_controls:]
            
            r2_score = model.score(X, y)
            print(f"  {self.features[self.state_idx[i]]}: R2 = {r2_score:.3f}")
        
        # 출력 행렬 정의
        self._define_output_matrix(n_states, n_controls)
        print("선형 시스템 동역학 학습 완료!")
    
    def _learn_dynamics_koopman(self, train_data):
        """Koopman 연산자를 사용한 시스템 동역학 학습"""
        
        # 데이터 추출
        X_state = train_data[:-1, self.state_idx]
        X_control = train_data[:-1, self.control_idx]
        X_dist = train_data[:-1, self.disturbance_idx]
        Y_next = train_data[1:, self.state_idx]
        
        n_states = len(self.state_idx)
        n_controls = len(self.control_idx)
        n_dist = len(self.disturbance_idx)
        
        # Lifting functions 정의 (다항식 기저)
        poly_state = PolynomialFeatures(degree=self.koopman_order, include_bias=False)
        
        # 상태를 고차원으로 변환
        X_state_lifted = poly_state.fit_transform(X_state)
        Y_next_lifted = poly_state.transform(Y_next)
        
        self.koopman_transform = poly_state
        self.lifting_dim = X_state_lifted.shape[1]
        
        print(f"  원래 상태 차원: {n_states}")
        print(f"  Lifting 후 차원: {self.lifting_dim}")
        
        # Koopman 행렬 학습 (고차원 선형 모델)
        self.A_lifted = np.zeros((self.lifting_dim, self.lifting_dim))
        self.B_lifted = np.zeros((self.lifting_dim, n_controls))
        self.E_lifted = np.zeros((self.lifting_dim, n_dist))
        
        for i in range(self.lifting_dim):
            X = np.hstack([X_state_lifted, X_control, X_dist])
            y = Y_next_lifted[:, i]
            
            model = LinearRegression()  # Ridge → LinearRegression
            model.fit(X, y)
            
            self.A_lifted[i, :] = model.coef_[:self.lifting_dim]
            self.B_lifted[i, :] = model.coef_[self.lifting_dim:self.lifting_dim+n_controls]
            self.E_lifted[i, :] = model.coef_[self.lifting_dim+n_controls:]
        
        # 원래 차원으로의 투영 행렬
        self.projection_matrix = np.zeros((n_states, self.lifting_dim))
        self.projection_matrix[:n_states, :n_states] = np.eye(n_states)
        
        # 원래 차원에서의 선형 근사
        self.A = self.projection_matrix @ self.A_lifted @ self.projection_matrix.T
        self.B = self.projection_matrix @ self.B_lifted
        self.E = self.projection_matrix @ self.E_lifted
        
        # 출력 행렬 정의
        self._define_output_matrix(n_states, n_controls)
        
        # 성능 평가
        Y_pred_lifted = X_state_lifted @ self.A_lifted.T + X_control @ self.B_lifted.T + X_dist @ self.E_lifted.T
        Y_pred = Y_pred_lifted[:, :n_states]
        
        for i in range(n_states):
            r2 = 1 - np.sum((Y_next[:, i] - Y_pred[:, i])**2) / np.sum((Y_next[:, i] - Y_next[:, i].mean())**2)
            print(f"  {self.features[self.state_idx[i]]}: R2 = {r2:.3f}")
        
        print("Koopman 시스템 동역학 학습 완료!")
            
    def _define_output_matrix(self, n_states, n_controls):
        """출력 행렬 C와 D 정의"""
        
        if self.observable_states == 'all':
            # 모든 상태가 측정 가능 (현재 가정)
            self.C = np.eye(n_states)
            self.D = np.zeros((n_states, n_controls))
            print("  출력 행렬: 모든 상태 측정 가능 (완전 관측)")
            
        else:
            # 일부 상태만 측정 가능
            observable_indices = self.observable_states
            n_outputs = len(observable_indices)
            
            self.C = np.zeros((n_outputs, n_states))
            for i, state_idx in enumerate(observable_indices):
                self.C[i, state_idx] = 1.0
            
            self.D = np.zeros((n_outputs, n_controls))
            print(f"  출력 행렬: {n_outputs}개 상태만 측정 가능 (부분 관측)")
    
    def get_measurement(self, state):
        """실제 측정값 계산 (출력 방정식 적용)"""
        if self.C is not None:
            return self.C @ state
        else:
            return state
    
    def _solve_mpc_step(self, current_state, current_disturbance, last_control=None):
        """MPC 최적화 문제 해결"""
        N = self.prediction_horizon
        M = self.control_horizon
        
        if self.dynamics_method == 'koopman' and self.A_lifted is not None:
            return self._solve_mpc_koopman(current_state, current_disturbance, last_control)
        else:
            return self._solve_mpc_linear(current_state, current_disturbance, last_control)
    
    def _solve_mpc_linear(self, current_state, current_disturbance, last_control=None):
        """선형 MPC 최적화"""
        N = self.prediction_horizon
        M = self.control_horizon
        n_states = len(self.state_idx)
        n_controls = len(self.control_idx)
        
        # 최적화 변수
        X = cp.Variable((N+1, n_states))
        U = cp.Variable((M, n_controls))
        
        # 목적함수
        obj = 0
        for t in range(M):
            if len(self.control_idx) >= 3:
                power = U[t, 1] + U[t, 2]
                obj += self.weights['power'] * power
            
            if t < N and len(self.state_idx) >= 4:
                temp_dev = cp.square(X[t, 3])
                obj += self.weights['comfort'] * temp_dev
            
            if last_control is not None and t == 0:
                control_change = cp.sum_squares(U[t, :] - last_control)
                obj += self.weights['control_change'] * control_change
                
            elif t > 0:
                control_change = cp.sum_squares(U[t, :] - U[t-1, :])
                obj += self.weights['control_change'] * control_change
        
        # 제약조건
        constraints = []
        
        # 초기 조건
        constraints.append(X[0, :] == current_state)
        
        # 시스템 동역학 (E 행렬 사용)
        for t in range(N):
            if t < M:
                next_state = self.A @ X[t, :] + self.B @ U[t, :] + self.E @ current_disturbance
            else:
                next_state = self.A @ X[t, :] + self.B @ U[M-1, :] + self.E @ current_disturbance
            constraints.append(X[t+1, :] == next_state)
        
        # 상태 제약
        if len(self.state_idx) >= 3:
            constraints.append(X[:, 2] <= self.constraints['co2_limit'])
            constraints.append(X[:, 0] <= self.constraints['pm10_limit'])
        
        if len(self.state_idx) >= 4:
            constraints.append(X[:, 3] >= self.constraints['temp_min'])
            constraints.append(X[:, 3] <= self.constraints['temp_max'])
        
        # 제어 제약
        constraints.append(U >= 0)
        constraints.append(U <= 1)
        
        # 최적화 문제 해결
        prob = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return U.value[0, :], X.value, True
        except:
            pass
        
        return np.ones(n_controls) * 0.5, None, False
    
    def _solve_mpc_koopman(self, current_state, current_disturbance, last_control=None):
        """Koopman 기반 MPC 최적화"""
        N = self.prediction_horizon
        M = self.control_horizon
        n_controls = len(self.control_idx)
        n_states = len(self.state_idx)
        
        # 현재 상태를 Koopman 공간으로 변환
        current_state_lifted = self.koopman_transform.transform(current_state.reshape(1, -1)).flatten()
        
        # 최적화 변수
        Z = cp.Variable((N+1, self.lifting_dim))
        U = cp.Variable((M, n_controls))
        X = cp.Variable((N+1, n_states))
        
        # 목적함수
        obj = 0
        for t in range(M):
            if len(self.control_idx) >= 3:
                power = U[t, 1] + U[t, 2]
                obj += self.weights['power'] * power
            
            if t < N:
                temp_dev = cp.square(X[t, 3])
                obj += self.weights['comfort'] * temp_dev
            
            if last_control is not None and t == 0:
                control_change = cp.sum_squares(U[t, :] - last_control)
                obj += self.weights['control_change'] * control_change
                
            elif t > 0:
                control_change = cp.sum_squares(U[t, :] - U[t-1, :])
                obj += self.weights['control_change'] * control_change
        
        # 제약조건
        constraints = []
        
        # 초기 조건
        constraints.append(Z[0, :] == current_state_lifted)
        
        # Koopman 동역학
        for t in range(N):
            if t < M:
                next_z = self.A_lifted @ Z[t, :] + self.B_lifted @ U[t, :] + self.E_lifted @ current_disturbance
            else:
                next_z = self.A_lifted @ Z[t, :] + self.B_lifted @ U[M-1, :] + self.E_lifted @ current_disturbance
            constraints.append(Z[t+1, :] == next_z)
        
        # 투영 제약
        for t in range(N+1):
            constraints.append(X[t, :] == Z[t, :n_states])
        
        # 상태 제약
        if len(self.state_idx) >= 3:
            constraints.append(X[:, 2] <= self.constraints['co2_limit'])
            constraints.append(X[:, 0] <= self.constraints['pm10_limit'])
        
        if len(self.state_idx) >= 4:
            constraints.append(X[:, 3] >= self.constraints['temp_min'])
            constraints.append(X[:, 3] <= self.constraints['temp_max'])
        
        # 제어 제약
        constraints.append(U >= 0)
        constraints.append(U <= 1)
        
        # 최적화
        prob = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return U.value[0, :], X.value, True
        except:
            pass
        
        return np.ones(n_controls) * 0.5, None, False
    
    def simulate(self, test_data, use_mpc=True):
        """시뮬레이션 실행 (출력 행렬 적용)"""
        n_steps = min(len(test_data) - self.prediction_horizon, 100)
        
        results = {
            'states': [],
            'measurements': [],  # 추가: 실제 측정값
            'controls': [],
            'power': [],
            'violations': [],
            'mpc_status': []
        }
        
        last_control = None
        estimated_state = None  # 추정 상태 (부분 관측시 필요)
        
        print(f"\n{'MPC' if use_mpc else '기준'} 시뮬레이션 실행 중 ({n_steps} 스텝)...")
        
        for t in range(n_steps):
            # 1. 측정값 획득 (출력 행렬 적용)
            if self.observable_states == 'all':
                # 완전 관측: 모든 상태 측정 가능
                current_state = test_data[t, self.state_idx]
                current_measurement = current_state  # y = x
            else:
                # 부분 관측: 일부만 측정 가능
                true_state = test_data[t, self.state_idx]  # 실제 상태 (시뮬레이션용)
                current_measurement = self.C @ true_state  # y = C*x
                
                # 상태 추정 필요 (간단한 방법: 이전 예측값 + 측정값 보정)
                if estimated_state is None:
                    # 초기: 측정값으로부터 역추정
                    estimated_state = self._estimate_initial_state(current_measurement)
                else:
                    # 예측-보정
                    estimated_state = self._estimate_state(
                        estimated_state, 
                        current_measurement, 
                        last_control,
                        test_data[t-1, self.disturbance_idx] if t > 0 else test_data[t, self.disturbance_idx]
                    )
                
                current_state = estimated_state
            
            current_dist = test_data[t, self.disturbance_idx]
            
            # 2. 제어 결정
            if use_mpc and self.A is not None:
                control, trajectory, success = self._solve_mpc_step(
                    current_state,  # 완전 관측 또는 추정된 상태
                    current_dist,
                    last_control
                )
                last_control = control
                results['mpc_status'].append(success)
            else:
                control = test_data[t, self.control_idx]
                results['mpc_status'].append(None)
            
            # 3. 다음 상태 예측
            if self.A is not None:
                next_state = self.A @ current_state + self.B @ control + self.E @ current_dist
                if self.observable_states != 'all':
                    estimated_state = next_state  # 다음 추정값 저장
            else:
                next_state = test_data[t+1, self.state_idx] if t+1 < len(test_data) else current_state
            
            # 4. 전력 계산
            if len(self.control_idx) >= 3:
                power = control[1] + control[2] if use_mpc else test_data[t, self.power_idx].sum()
            else:
                power = 0
            
            # 5. 제약 위반 체크
            violations = {}
            if len(self.state_idx) >= 3:
                violations['co2'] = next_state[2] > self.constraints['co2_limit']
                violations['pm10'] = next_state[0] > self.constraints['pm10_limit']
            if len(self.state_idx) >= 4:
                violations['temp'] = (next_state[3] < self.constraints['temp_min'] or 
                                    next_state[3] > self.constraints['temp_max'])
            
            # 6. 결과 저장
            results['states'].append(next_state)
            results['measurements'].append(current_measurement)  # 측정값도 저장
            results['controls'].append(control)
            results['power'].append(power)
            results['violations'].append(violations)
        
        return results
    
    def _estimate_initial_state(self, measurement):
        """초기 상태 추정 (부분 관측시)"""
        n_states = len(self.state_idx)
        
        if self.observable_states == 'all':
            return measurement
        
        # 간단한 방법: 측정된 값 + 기본값
        initial_state = np.zeros(n_states)
        
        # 측정 가능한 상태는 측정값 사용
        for i, obs_idx in enumerate(self.observable_states):
            initial_state[obs_idx] = measurement[i]
        
        # 측정 불가능한 상태는 기본값 (0) 또는 추정
        # 예: PM 센서가 없으면 실외 PM의 절반으로 추정
        
        return initial_state

    def _estimate_state(self, prev_estimate, measurement, control, disturbance):
        """상태 추정기 (간단한 예측-보정)"""
        
        # 1. 예측 단계
        predicted_state = self.A @ prev_estimate + self.B @ control + self.E @ disturbance
        
        # 2. 보정 단계 (측정값 활용)
        predicted_measurement = self.C @ predicted_state
        measurement_error = measurement - predicted_measurement
        
        # 간단한 보정 (Kalman gain 대신 고정 gain 사용)
        correction_gain = 0.5
        
        # 측정 가능한 상태만 보정
        corrected_state = predicted_state.copy()
        for i, obs_idx in enumerate(self.observable_states):
            if obs_idx < len(corrected_state):
                corrected_state[obs_idx] += correction_gain * measurement_error[i]
        
        return corrected_state
        
    def evaluate(self, mpc_results, baseline_results, test_data=None):
        """MPC 성능 평가"""
        print("\n성능 평가 중...")
        
        # 전력 소비 비교
        mpc_power = np.array(mpc_results['power'])
        baseline_power = np.array(baseline_results['power'])
        
        power_metrics = {
            'mpc_total': np.sum(mpc_power),
            'baseline_total': np.sum(baseline_power),
            'reduction_percent': (1 - np.sum(mpc_power)/(np.sum(baseline_power) + 1e-8)) * 100,
            'mpc_mean': np.mean(mpc_power),
            'baseline_mean': np.mean(baseline_power)
        }
        
        # 제약 만족도
        mpc_viol = mpc_results['violations']
        baseline_viol = baseline_results['violations']
        
        constraint_metrics = {}
        for key in ['co2', 'pm10', 'temp']:
            if mpc_viol and key in mpc_viol[0]:
                constraint_metrics[f'mpc_{key}_violations'] = sum(v.get(key, False) for v in mpc_viol)
                constraint_metrics[f'baseline_{key}_violations'] = sum(v.get(key, False) for v in baseline_viol)
        
        # 제어 안정성
        mpc_controls = np.array(mpc_results['controls'])
        baseline_controls = np.array(baseline_results['controls'])
        
        mpc_changes = np.sum(np.abs(np.diff(mpc_controls, axis=0)))
        baseline_changes = np.sum(np.abs(np.diff(baseline_controls, axis=0)))
        
        stability_metrics = {
            'mpc_changes': mpc_changes,
            'baseline_changes': baseline_changes,
            'improvement': (1 - mpc_changes/(baseline_changes + 1e-8)) * 100
        }
        
        # MPC 성공률
        if mpc_results['mpc_status']:
            success_rate = sum(s for s in mpc_results['mpc_status'] if s is not None) / len(mpc_results['mpc_status'])
        else:
            success_rate = 0
        
        self.evaluation_results = {
            'power': power_metrics,
            'constraints': constraint_metrics,
            'stability': stability_metrics,
            'mpc_success_rate': success_rate
        }
        
        return self.evaluation_results
    
    def print_report(self):
        """평가 결과 출력"""
        if not self.evaluation_results:
            print("평가 결과가 없습니다.")
            return
        
        print("\n" + "="*60)
        print("MPC 성능 평가 보고서")
        print("="*60)
        
        print("\n1. 전력 소비")
        power = self.evaluation_results['power']
        print(f"  MPC 총 전력: {power['mpc_total']:.2f}")
        print(f"  기준 총 전력: {power['baseline_total']:.2f}")
        print(f"  절감율: {power['reduction_percent']:.1f}%")
        
        print("\n2. 제약 만족도")
        constraints = self.evaluation_results['constraints']
        for key in ['co2', 'pm10', 'temp']:
            mpc_key = f'mpc_{key}_violations'
            baseline_key = f'baseline_{key}_violations'
            if mpc_key in constraints:
                print(f"  {key.upper()}: MPC {constraints[mpc_key]} vs 기준 {constraints[baseline_key]}")
        
        print("\n3. 제어 안정성")
        stability = self.evaluation_results['stability']
        print(f"  제어 변화량: MPC {stability['mpc_changes']:.2f} vs 기준 {stability['baseline_changes']:.2f}")
        print(f"  개선율: {stability['improvement']:.1f}%")
        
        print(f"\n4. MPC 최적화 성공률: {self.evaluation_results['mpc_success_rate']*100:.1f}%")