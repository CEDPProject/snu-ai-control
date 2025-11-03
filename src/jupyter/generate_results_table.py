import json
import pandas as pd
from pathlib import Path
import numpy as np

# 데이터 경로 설정
base_path = Path('../saved_model/version2_250813_processed')
rooms = {
    'livingroom': '거실',
    'room1': '방 1',
    'room2': '방 2'
}
models = ['DLinear', 'FreTS', 'iTransformer', 'Koopa', 'MICN', 
          'PatchTST', 'TimeMixer', 'TimesNet', 'TimeXer', 'TSMixer']
systems = {
    'KETI_system_control': '환기유효전력(W)',
    'system_control': '냉난방유효전력(W)'
}

def load_test_results(room, model, system):
    """test_results.json 파일 로드"""
    file_path = base_path / room / model / system / 'version0' / 'test_results.json'
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"오류 발생 ({file_path}): {e}")
        return None

# 모든 데이터 수집
all_results = []

for room_key, room_name in rooms.items():
    for model in models:
        for system_key, system_name in systems.items():
            result = load_test_results(room_key, model, system_key)
            if result:
                all_results.append({
                    '공간': room_name,
                    '시스템': system_name,
                    '모델': model,
                    '전력량 정확도': result.get('prediction_accuracy', None),
                    'MSE': result.get('mse', None),
                    'MAE': result.get('mae', None),
                    'RMSE': result.get('rmse', None),
                    'MAPE': result.get('mape', None)
                })

# DataFrame 생성
df = pd.DataFrame(all_results)

print(f"총 {len(df)}개의 결과 로드됨\n")

# 1. CSV 파일로 저장 (전체 데이터)
csv_filename = 'model_results_all.csv'
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
print(f"✓ 전체 결과가 '{csv_filename}'에 저장되었습니다.")

# 2. 엑셀 파일로 저장 (이미지와 같은 형식)
excel_filename = 'model_results_formatted.xlsx'

with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    # 각 시스템별로 시트 생성
    for system_name in systems.values():
        system_data = df[df['시스템'] == system_name].copy()
        
        # 공간별로 데이터 재구성
        result_dfs = []
        for room_name in rooms.values():
            room_data = system_data[system_data['공간'] == room_name][['모델', '전력량 정확도', 'MSE']].copy()
            room_data = room_data.sort_values('모델').reset_index(drop=True)
            room_data.columns = [f'{room_name} - {col}' if col != '모델' else col for col in room_data.columns]
            result_dfs.append(room_data)
        
        # 모든 공간의 데이터를 하나로 합치기
        if result_dfs:
            combined_df = result_dfs[0]
            for i in range(1, len(result_dfs)):
                combined_df = pd.merge(combined_df, result_dfs[i], on='모델', how='outer')
            
            # 시트 이름 (환기유효전력 또는 냉난방유효전력)
            sheet_name = system_name.replace('(W)', '').strip()
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"✓ '{sheet_name}' 시트 생성됨")

print(f"\n✓ 형식화된 결과가 '{excel_filename}'에 저장되었습니다.")

# 3. 이미지와 동일한 형식으로 개별 CSV 파일 생성
output_dir = Path('results_tables')
output_dir.mkdir(exist_ok=True)

for system_name in systems.values():
    for room_name in rooms.values():
        # 해당 공간과 시스템의 데이터 필터링
        filtered_data = df[(df['공간'] == room_name) & (df['시스템'] == system_name)].copy()
        
        # 필요한 컬럼만 선택하고 정렬
        table_data = filtered_data[['모델', '전력량 정확도', 'MSE']].sort_values('모델').reset_index(drop=True)
        
        # 파일명 생성
        filename = f"{room_name}_{system_name.replace('(W)', '').strip()}.csv"
        filepath = output_dir / filename
        
        # CSV로 저장
        table_data.to_csv(filepath, index=False, encoding='utf-8-sig')

print(f"\n✓ 개별 테이블이 '{output_dir}' 폴더에 저장되었습니다.")

# 4. 이미지와 동일한 레이아웃으로 하나의 엑셀 파일 생성
excel_layout_filename = 'model_results_layout.xlsx'

with pd.ExcelWriter(excel_layout_filename, engine='openpyxl') as writer:
    workbook = writer.book
    
    for system_key, system_name in systems.items():
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils.dataframe import dataframe_to_rows
        
        # 새 워크시트 생성
        sheet_name = system_name.replace('(W)', '').strip()
        worksheet = workbook.create_sheet(title=sheet_name)
        
        # 헤더 스타일
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        
        # 데이터 스타일
        cell_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # 각 공간에 대한 테이블 생성 (가로로 배치)
        start_col = 1
        row_offset = 1
        
        for room_key, room_name in rooms.items():
            # 해당 공간의 데이터 가져오기
            room_data = df[(df['공간'] == room_name) & (df['시스템'] == system_name)].copy()
            table_data = room_data[['모델', '전력량 정확도', 'MSE']].sort_values('모델').reset_index(drop=True)
            
            # 헤더 추가
            header = [f"{room_name} ({system_name})", "", ""]
            worksheet.cell(row=row_offset, column=start_col, value=header[0])
            worksheet.merge_cells(start_row=row_offset, start_column=start_col, 
                                 end_row=row_offset, end_column=start_col+2)
            
            # 헤더 스타일 적용
            cell = worksheet.cell(row=row_offset, column=start_col)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            
            # 컬럼 헤더
            row_offset += 1
            col_headers = ['모델', '전력량 정확도', 'MSE']
            for col_idx, header_text in enumerate(col_headers, start=start_col):
                cell = worksheet.cell(row=row_offset, column=col_idx, value=header_text)
                cell.fill = cell_fill
                cell.font = Font(bold=True)
                cell.border = border
                cell.alignment = Alignment(horizontal='center')
            
            # 데이터 추가
            row_offset += 1
            for row_idx, row_data in enumerate(dataframe_to_rows(table_data, index=False, header=False), start=row_offset):
                for col_idx, value in enumerate(row_data, start=start_col):
                    cell = worksheet.cell(row=row_idx, column=col_idx, value=value)
                    cell.border = border
                    
                    # 숫자 포맷 적용
                    if col_idx > start_col:  # 모델명이 아닌 경우
                        if isinstance(value, (int, float)):
                            cell.number_format = '0.0000'
                    
                    cell.alignment = Alignment(horizontal='center')
            
            # 다음 테이블 위치
            start_col += 4  # 3개 컬럼 + 1개 빈 컬럼
            
            # 컬럼 너비 조정
            for col in range(start_col-4, start_col-1):
                worksheet.column_dimensions[chr(64+col)].width = 15
        
        # 첫 번째 빈 시트 삭제
        if 'Sheet' in workbook.sheetnames:
            del workbook['Sheet']

workbook.save(excel_layout_filename)
print(f"\n✓ 레이아웃 형식의 결과가 '{excel_layout_filename}'에 저장되었습니다.")

print("\n" + "="*60)
print("모든 파일 생성 완료!")
print("="*60)
print(f"1. {csv_filename} - 전체 데이터 (CSV)")
print(f"2. {excel_filename} - 시스템별 시트 (Excel)")
print(f"3. {excel_layout_filename} - 이미지 레이아웃 형식 (Excel)")
print(f"4. {output_dir}/ - 개별 테이블 (CSV)")
