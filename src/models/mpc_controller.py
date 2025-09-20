# models/mpc_model.py
import numpy as np
import cvxpy as cp
from sklearn.linear_model import LinearRegression
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
        
        # 모델 파라미터
        self.A = None
        self.B = None
        self.C = None
        
        # Koopman 관련
        self.koopman_transform = None
        self.A_lifted = None
        self.B_lifted = None
        self.C_lifted = None
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
        """시스템 동역학 학습"""
        print("\n시스템 동역학 학습 시작...")
        
        # 데이터 추출
        X_state = train_data[:-1, self.state_idx]
        X_control = train_data[:-1, self.control_idx]
        X_dist = train_data[:-1, self.disturbance_idx]
        Y_next = train_data[1:, self.state_idx]
        
        n_states = len(self.state_idx)
        n_controls = len(self.control_idx)
        n_dist = len(self.disturbance_idx)
        
        # 시스템 행렬 학습
        self.A = np.zeros((n_states, n_states))
        self.B = np.zeros((n_states, n_controls))
        self.C = np.zeros((n_states, n_dist))
        
        for i in range(n_states):
            X = np.hstack([X_state, X_control, X_dist])
            y = Y_next[:, i]
            
            model = LinearRegression()
            model.fit(X, y)
            
            self.A[i, :] = model.coef_[:n_states]
            self.B[i, :] = model.coef_[n_states:n_states+n_controls]
            self.C[i, :] = model.coef_[n_states+n_controls:]
            
            r2_score = model.score(X, y)
            print(f"  {self.features[self.state_idx[i]]}: R2 = {r2_score:.3f}")
        
        print("시스템 동역학 학습 완료!")
        
    def _solve_mpc_step(self, current_state, current_disturbance, last_control=None):
        """한 스텝의 MPC 최적화 문제 해결"""
        N = self.prediction_horizon
        M = self.control_horizon
        n_states = len(self.state_idx)
        n_controls = len(self.control_idx)
        
        # 최적화 변수
        X = cp.Variable((N+1, n_states))
        U = cp.Variable((M, n_controls))
        
        # 목적함수 구성
        obj = 0
        breakpoint()
        # 전력 최소화
        for t in range(M):
            if len(self.control_idx) >= 3:
                power = U[t, 1] + U[t, 2]  # 환기전원 + 냉난방전원
                obj += self.weights['power'] * power
            
            # 온도 편안함
            if t < N and len(self.state_idx) >= 4:
                temp_dev = cp.square(X[t, 3])  # 온도 편차
                obj += self.weights['comfort'] * temp_dev
            
            # 제어 변화 페널티
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
        
        # 시스템 동역학
        for t in range(N):
            if t < M:
                next_state = self.A @ X[t, :] + self.B @ U[t, :] + self.C @ current_disturbance
            else:
                next_state = self.A @ X[t, :] + self.B @ U[M-1, :] + self.C @ current_disturbance
            constraints.append(X[t+1, :] == next_state)
        
        # 상태 제약
        if len(self.state_idx) >= 3:
            constraints.append(X[:, 2] <= self.constraints['co2_limit'])  # CO2
            constraints.append(X[:, 0] <= self.constraints['pm10_limit'])  # PM10
        
        if len(self.state_idx) >= 4:
            constraints.append(X[:, 3] >= self.constraints['temp_min'])  # 온도 하한
            constraints.append(X[:, 3] <= self.constraints['temp_max'])  # 온도 상한
        
        # 제어 제약
        constraints.append(U >= 0)
        constraints.append(U <= 1)
        
        # 최적화 문제 해결
        prob = cp.Problem(cp.Minimize(obj), constraints)
        
        try:
            prob.solve(solver=cp.OSQP, verbose=False, warm_start=True)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return U.value[0, :], X.value, True
        except Exception as e:
            pass
        
        # 실패시 기본값 반환
        return np.ones(n_controls) * 0.5, None, False
    
    def simulate(self, test_data, use_mpc=True):
        """시뮬레이션 실행"""
        n_steps = min(len(test_data) - self.prediction_horizon, 100)  # 최대 100 스텝
        
        results = {
            'states': [],
            'controls': [],
            'power': [],
            'violations': [],
            'mpc_status': []
        }
        
        last_control = None
        
        print(f"\n{'MPC' if use_mpc else '기준'} 시뮬레이션 실행 중 ({n_steps} 스텝)...")
        
        for t in range(n_steps):
            current_state = test_data[t, self.state_idx]
            current_dist = test_data[t, self.disturbance_idx]
            
            if use_mpc and self.A is not None:
                # MPC로 제어 결정
                control, trajectory, success = self._solve_mpc_step(
                    current_state, 
                    current_dist,
                    last_control
                )
                last_control = control
                results['mpc_status'].append(success)
            else:
                # 기존 데이터의 제어값 사용
                control = test_data[t, self.control_idx]
                results['mpc_status'].append(None)
            
            # 다음 상태 예측
            if self.A is not None:
                next_state = self.A @ current_state + self.B @ control + self.C @ current_dist
            else:
                next_state = test_data[t+1, self.state_idx] if t+1 < len(test_data) else current_state
            
            # 전력 계산
            if len(self.control_idx) >= 3:
                power = control[1] + control[2] if use_mpc else test_data[t, self.power_idx].sum()
            else:
                power = 0
            
            # 제약 위반 체크
            violations = {}
            if len(self.state_idx) >= 3:
                violations['co2'] = next_state[2] > self.constraints['co2_limit']
                violations['pm10'] = next_state[0] > self.constraints['pm10_limit']
            if len(self.state_idx) >= 4:
                violations['temp'] = (next_state[3] < self.constraints['temp_min'] or 
                                     next_state[3] > self.constraints['temp_max'])
            
            results['states'].append(next_state)
            results['controls'].append(control)
            results['power'].append(power)
            results['violations'].append(violations)
        
        return results
    
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
