import numpy as np
from scipy.spatial.distance import pdist  # 모든 점 쌍 간 거리 계산을 위해 사용
import os
import pandas as pd


# GSLIB .dat 파일 로드 함수
# 이 함수는 파일 경로를 입력받아, 주석을 제외한 숫자 데이터를 numpy 배열로 반환합니다.
def load_numeric_dat_file(filepath):
    """
    숫자로만 구성된 .dat 파일을 읽어 numpy 배열로 반환합니다.
    주석 처리된 라인 (예: '#' 또는 '!'로 시작)은 무시합니다.
    파일을 찾을 수 없거나 유효한 숫자 데이터가 없는 경우 오류를 발생시킵니다.
    """
    data_lines = []  # 데이터를 저장할 리스트 초기화
    try:
        with open(
            filepath, "r", encoding="utf-8"
        ) as f:  # 파일을 읽기 모드로 엽니다. 인코딩은 utf-8로 가정합니다.
            for line in f:  # 파일의 각 줄에 대해 반복합니다.
                stripped_line = (
                    line.strip()
                )  # 현재 줄의 앞뒤 공백(줄바꿈 문자 등)을 제거합니다.
                # 줄이 비어있지 않고, 주석 문자로 시작하지 않는 경우에만 데이터를 처리합니다.
                if stripped_line and not stripped_line.startswith(("#", "!")):
                    try:
                        # 공백을 기준으로 줄을 나누고, 각 항목을 float(실수)형으로 변환하여 리스트에 추가합니다.
                        data_lines.append(list(map(float, stripped_line.split())))
                    except ValueError:
                        # 만약 줄의 내용 중 float으로 변환할 수 없는 값이 있다면 경고 메시지를 출력하고 해당 줄은 건너뜁니다.
                        print(
                            f"Warning: Could not convert line to float, skipping: {stripped_line}"
                        )

        if (
            not data_lines
        ):  # 모든 줄을 처리한 후에도 유효한 데이터가 없다면 오류를 발생시킵니다.
            raise ValueError(f"No valid numeric data found in {filepath}")
        return np.array(
            data_lines
        )  # 최종적으로 데이터 리스트를 numpy 배열로 변환하여 반환합니다.
    except FileNotFoundError:  # 지정된 파일을 찾을 수 없는 경우 발생하는 오류입니다.
        raise FileNotFoundError(f"Error: Data file not found at '{filepath}'")
    except Exception as e:  # 그 외 다른 예외 발생 시 처리합니다.
        raise Exception(f"An error occurred while loading data from '{filepath}': {e}")


# 좌표로부터 최대 점 간 거리 계산 함수
# 이 함수는 X, Y 좌표 배열을 입력받아, 모든 점 쌍 간의 유클리드 거리 중 최대값을 계산합니다.
# 이 값은 반변량도 분석 시 최대 분석 거리(max_lag_dist_practical)를 설정하는 데 사용됩니다.
def calculate_max_distance(x_coords, y_coords):
    """
    좌표로부터 최대 점 간 거리를 계산합니다.
    데이터 포인트가 2개 미만이거나 유효하지 않은 경우 기본값 1.0을 반환합니다.
    """
    # 입력된 좌표 데이터의 유효성을 검사합니다. 최소 2개의 포인트가 필요합니다.
    if x_coords is None or y_coords is None or len(x_coords) < 2 or len(y_coords) < 2:
        print(
            "Warning: Not enough data points (<2) to calculate max distance. Returning default 1.0"
        )
        return 1.0  # 기본값 반환

    # X와 Y 좌표를 (N, 2) 형태의 numpy 배열로 결합합니다 (N은 포인트 수).
    points = np.vstack((x_coords, y_coords)).T
    if points.shape[0] < 2:  # 결합 후 포인트 수 재확인
        print(
            "Warning: Less than 2 valid points after stacking for max distance. Returning default 1.0"
        )
        return 1.0

    try:
        # scipy.spatial.distance.pdist를 사용하여 모든 점 쌍 간의 유클리드 거리를 계산합니다.
        # pdist는 거리 행렬의 상삼각(또는 하삼각) 부분을 1차원 배열로 반환합니다.
        distances = pdist(points)
        if len(distances) == 0:  # 거리가 계산되지 않은 경우 (예: 모든 점이 동일 위치)
            print(
                "Warning: No pairwise distances calculated (points might be identical). Returning default 1.0 for max distance."
            )
            return 1.0

        max_dist = np.max(distances)  # 계산된 거리 중 최대값을 찾습니다.

        # 최대 거리가 유효하지 않은 값(NaN)이거나 0 또는 매우 작은 경우 기본값으로 대체합니다.
        # 이는 모든 점이 한 점에 모여있거나 계산 오류 발생 시를 대비한 것입니다.
        if pd.isna(max_dist) or max_dist <= 1e-9:
            max_dist = 1.0
            print(
                f"Warning: Calculated maximum pairwise distance was NaN, zero, or too small. Reset to {max_dist}."
            )
        return max_dist
    except Exception as e:  # 거리 계산 중 예외 발생 시 처리합니다.
        print(
            f"Error calculating maximum pairwise distance: {e}. Returning default 1.0"
        )
        return 1.0


# GSLIB .par 파일 내용 생성 함수
# 이 함수는 GSLIB의 GAMV 프로그램 실행에 필요한 다양한 파라미터들을 입력받아,
# .par 파일 형식에 맞는 문자열 리스트를 생성하여 반환합니다.
def create_gslib_par_content(
    data_file_for_gslib_par,  # GSLIB이 읽을 데이터 파일 경로
    x_col_num,  # X 좌표 열 번호 (1-based)
    y_col_num,  # Y 좌표 열 번호 (1-based)
    z_col_num,  # Z 좌표 열 번호 (1-based, 2D 분석 시 0)
    num_vars_to_activate,  # .par 파일 내에서 활성화할 변수의 수 (보통 1)
    col_of_var_to_activate,  # 분석 대상 변수가 데이터 파일 내 몇 번째 열에 있는지 (1-based)
    trim_min,  # 데이터 트리밍 하한값 (이 값 미만은 무시)
    trim_max,  # 데이터 트리밍 상한값 (이 값 초과는 무시)
    gslib_output_filename,  # GAMV 실행 결과가 저장될 .out 파일명
    nlag,  # 계산할 Lag의 개수
    xlag,  # 단위 Lag 거리 (Lag separation distance)
    xtol,  # Lag 허용 오차 (Lag tolerance, 보통 xlag의 절반)
    num_directions,  # 분석할 방향의 수
    direction_lines,  # 각 방향별 설정 문자열 리스트 (azm, atol, bandh, dip, dtol, bandv)
    standardize_sill,  # Sill 값을 1로 표준화할지 여부 (0=아니오, 1=예)
    nvarg,  # 계산할 반변량도(쌍)의 개수
    variogram_details_lines,  # 각 반변량도 쌍에 대한 상세 정보 튜플들의 리스트 (tail, head, ivtype, [cutoff])
):
    """
    GSLIB .par 파일 내용을 생성합니다.
    """
    lines = [
        "                  Parameters for GAMV",  # .par 파일의 제목 줄
        "                  *******************",
        "",  # 빈 줄
        "START OF PARAMETERS:",  # 파라미터 시작을 알리는 GSLIB 키워드
        f"{data_file_for_gslib_par}",  # 1. 데이터 파일 경로
        f"{int(x_col_num)}  {int(y_col_num)}  {int(z_col_num)}",  # 2. X, Y, Z 좌표 컬럼 번호
        f"{int(num_vars_to_activate)}  {int(col_of_var_to_activate)}",  # 3. 분석할 변수 개수 및 해당 변수의 컬럼 번호
        f"{trim_min}  {trim_max}",  # 4. 트리밍 한계
        f"{gslib_output_filename}",  # 5. 결과 .out 파일명
        f"{int(nlag)}",  # 6. Lag 개수
        f"{xlag:.4f}",  # 7. 단위 Lag 거리 (소수점 4자리까지 표기)
        f"{xtol:.4f}",  # 8. Lag 허용 오차 (소수점 4자리까지 표기)
        f"{int(num_directions)}",  # 9. 분석할 방향 수
    ]
    lines.extend(direction_lines)  # 10. 각 방향별 설정 라인들 추가
    lines.append(f"{int(standardize_sill)}")  # 11. Sill 표준화 여부
    lines.append(f"{int(nvarg)}")  # 12. 계산할 반변량도(쌍)의 개수

    # 13. 각 반변량도 쌍에 대한 상세 정보 추가
    for detail_tuple in variogram_details_lines:
        if len(detail_tuple) == 4:  # 지시자 반변량도 (tail, head, ivtype, cutoff_value)
            lines.append(
                # 각 항목을 정해진 너비에 맞춰 정렬하여 GSLIB이 잘 인식하도록 함
                f"{int(detail_tuple[0]):<2}  {int(detail_tuple[1]):<2}  {int(detail_tuple[2]):<3} {detail_tuple[3]}"
            )
        elif len(detail_tuple) == 3:  # 일반 반변량도 (tail, head, ivtype)
            lines.append(
                f"{int(detail_tuple[0]):<2}  {int(detail_tuple[1]):<2}  {int(detail_tuple[2]):<3}"
            )
    return lines


# --- 스크립트 주요 실행 부분 ---
if __name__ == "__main__":
    # ==============================================================================
    # 사용자가 직접 설정해야 하는 부분 (하드코딩된 값들 유지)
    # ==============================================================================
    # 이 스크립트는 PYTHON_COORD_DATA_FILE에서 좌표 정보를 읽어 max_lag_dist_practical을 계산하고,
    # GSLIB_TARGET_DATA_FILE은 생성될 .par 파일 내에 기록될 데이터 파일 경로입니다.
    # 두 파일이 동일할 수도 있고, 다를 수도 있습니다 (예: 좌표만 있는 파일 vs 모든 변수 포함 파일)
    PYTHON_COORD_DATA_FILE = "jura259_class.dat"
    GSLIB_TARGET_DATA_FILE = (
        "jura259.dat"  # GSLIB이 사용할 실제 데이터 파일 (경로 주의)
    )

    # 생성될 .par 파일들이 저장될 디렉토리
    OUTPUT_PAR_DIR = "."  # 현재 디렉토리에 저장

    # 데이터 파일 내의 컬럼 번호 정의 (GSLIB은 1부터 시작하는 인덱스 사용)
    VARIABLE_COLUMN_INDICES_1_BASED = {
        "Xlocation": 1,
        "Ylocation": 2,
        "Rocktype": 3,
        "Landuse": 4,
        "Cd": 5,
        "Cu": 6,
        "Pb": 7,
        "Co": 8,
        "Cr": 9,
        "Ni": 10,
        "Zn": 11,
    }

    # .par 파일을 생성할 변수들과 그 특성 정의
    VARIABLES_TO_PROCESS = {
        "Rocktype": {"type": "categorical", "categories": 5, "ivtype_cat": 10},
        "Landuse": {"type": "categorical", "categories": 4, "ivtype_cat": 10},
        "Cd": {"type": "continuous", "ivtype_cont": 1},  # ivtype 1: Semivariogram
        "Cu": {"type": "continuous", "ivtype_cont": 1},
        "Pb": {"type": "continuous", "ivtype_cont": 1},
        "Co": {"type": "continuous", "ivtype_cont": 1},
        "Cr": {"type": "continuous", "ivtype_cont": 1},
        "Ni": {"type": "continuous", "ivtype_cont": 1},
        "Zn": {"type": "continuous", "ivtype_cont": 1},
    }

    # .par 파일 생성 시 사용할 단위 Lag 거리(xlag) 후보 값들의 리스트
    USER_DEFINED_XLAG_CANDIDATES = [0.0937, 0.1118, 0.15, 0.20, 0.25]

    # 계산될 Lag 개수(nlag)의 최소 및 최대 한계
    MIN_NLAG_LIMIT = 10
    MAX_NLAG_LIMIT = 40
    # ==============================================================================

    # USER_DEFINED_XLAG_CANDIDATES가 비어있는지 확인
    if not USER_DEFINED_XLAG_CANDIDATES:
        print(
            "CRITICAL ERROR: USER_DEFINED_XLAG_CANDIDATES list is empty. Cannot generate .par files."
        )
        exit()  # 프로그램 종료

    # OUTPUT_PAR_DIR이 현재 디렉토리가 아닌 경우, 해당 디렉토리 생성
    if OUTPUT_PAR_DIR != "." and not os.path.exists(OUTPUT_PAR_DIR):
        os.makedirs(OUTPUT_PAR_DIR, exist_ok=True)
        print(f"Output directory '{OUTPUT_PAR_DIR}' created.")

    # 좌표 데이터 로드 및 최대 분석 거리 계산 (nlag 결정용)
    max_lag_dist_practical = 0.0  # 기본값 초기화
    try:
        data_array_for_coords = load_numeric_dat_file(PYTHON_COORD_DATA_FILE)
        print(
            f"Coordinate data loaded from '{PYTHON_COORD_DATA_FILE}'. Shape: {data_array_for_coords.shape}"
        )

        x_col_idx_0_based = VARIABLE_COLUMN_INDICES_1_BASED["Xlocation"] - 1
        y_col_idx_0_based = VARIABLE_COLUMN_INDICES_1_BASED["Ylocation"] - 1

        if not (
            0 <= x_col_idx_0_based < data_array_for_coords.shape[1]
            and 0 <= y_col_idx_0_based < data_array_for_coords.shape[1]
        ):
            raise ValueError(
                f"Xlocation or Ylocation column index out of bounds for '{PYTHON_COORD_DATA_FILE}'. Check VARIABLE_COLUMN_INDICES_1_BASED."
            )

        x_coords_arr = data_array_for_coords[:, x_col_idx_0_based]
        y_coords_arr = data_array_for_coords[:, y_col_idx_0_based]

        max_dist_val = calculate_max_distance(x_coords_arr, y_coords_arr)
        max_lag_dist_practical = (
            max_dist_val / 2.0
        )  # 반변량도 분석 시 최대 거리는 보통 전체 데이터 영역 최대 거리의 절반 정도 사용
        print(
            f"\nReference Info (from '{PYTHON_COORD_DATA_FILE}' for lag parameter estimation):"
        )
        print(
            f"  Max Pairwise Distance in data / 2: {max_lag_dist_practical:.4f} (This will be the target max distance for variogram calculation)"
        )

    except FileNotFoundError:
        print(
            f"CRITICAL ERROR: Coordinate data file '{PYTHON_COORD_DATA_FILE}' not found. This file is essential for determining nlag. Aborting .par file generation."
        )
        exit()
    except ValueError as ve:
        print(
            f"CRITICAL ERROR: Problem with coordinate data file '{PYTHON_COORD_DATA_FILE}': {ve}. Aborting."
        )
        exit()
    except Exception as e:
        print(
            f"CRITICAL ERROR: An unexpected error occurred while processing coordinate data from '{PYTHON_COORD_DATA_FILE}': {e}. Aborting."
        )
        exit()

    # 사용할 xlag 값들 (numpy 배열로 변환)
    XLAG_VALUES_TO_USE = np.array(USER_DEFINED_XLAG_CANDIDATES)

    # XLAG_VALUES_TO_USE 유효성 재확인 (이론상 위에서 이미 처리됨)
    if XLAG_VALUES_TO_USE.size == 0:
        print(
            "Critical Error: No XLAG_VALUES available after processing. Aborting .par file creation."
        )
        exit()

    print(
        f"\nUser-defined xlag values to be used for generating .par files: {[float(f'{val:.4f}') for val in XLAG_VALUES_TO_USE]}"
    )
    print(
        f"Number of lags (nlag) will be calculated for each xlag to cover up to max_lag_dist_practical: {max_lag_dist_practical:.4f} (within limits: min={MIN_NLAG_LIMIT}, max={MAX_NLAG_LIMIT})"
    )

    # GSLIB .par 파일에 사용될 좌표 컬럼 번호 (1-based)
    X_COL_FOR_GSLIB_PAR = VARIABLE_COLUMN_INDICES_1_BASED["Xlocation"]
    Y_COL_FOR_GSLIB_PAR = VARIABLE_COLUMN_INDICES_1_BASED["Ylocation"]
    Z_COL_FOR_GSLIB_PAR = 0  # 2D 분석의 경우 Z 좌표 컬럼은 0 (사용 안 함)

    par_file_count = 0  # 생성된 .par 파일 개수 카운트

    # 각 변수 및 각 xlag 값에 대해 .par 파일 생성
    for var_name, var_info in VARIABLES_TO_PROCESS.items():
        if var_name not in VARIABLE_COLUMN_INDICES_1_BASED:
            print(
                f"Warning: Column index for variable '{var_name}' not defined in VARIABLE_COLUMN_INDICES_1_BASED. Skipping this variable."
            )
            continue

        actual_var_column_in_gslib_datafile = VARIABLE_COLUMN_INDICES_1_BASED[var_name]
        internal_active_var_idx = (
            1  # .par 파일 내에서는 활성화된 변수 목록 중 첫 번째를 의미
        )

        # 범주형 변수의 경우, 각 카테고리별로 .par 파일 생성
        if var_info["type"] == "categorical":
            num_categories = var_info["categories"]
            ivtype_for_categorical = var_info.get("ivtype_cat", 10)

            for cat_value in range(1, num_categories + 1):
                var_name_for_file = f"{var_name}_cat{cat_value}"  # 예: Rocktype_cat1

                for xlag_val in XLAG_VALUES_TO_USE:
                    if xlag_val <= 1e-9:  # xlag 값이 너무 작으면 건너뛰기
                        print(
                            f"Skipping generation for var='{var_name_for_file}', xlag={xlag_val:.4f} (xlag is too small)."
                        )
                        continue

                    nlag_to_use = MIN_NLAG_LIMIT
                    if max_lag_dist_practical > 1e-9 and xlag_val > 1e-9:
                        nlag_calculated_float = max_lag_dist_practical / xlag_val
                        nlag_to_use = int(round(nlag_calculated_float))
                        if nlag_to_use < MIN_NLAG_LIMIT:
                            nlag_to_use = MIN_NLAG_LIMIT
                        elif nlag_to_use > MAX_NLAG_LIMIT:
                            nlag_to_use = MAX_NLAG_LIMIT
                    else:
                        print(
                            f"  Warning: Using default nlag={nlag_to_use} for {var_name_for_file}, xlag={xlag_val:.4f} due to invalid max_lag_dist or xlag_val."
                        )

                    xtol_val = xlag_val / 2.0
                    if xtol_val <= 1e-9:
                        xtol_val = 1e-9

                    # GSLIB .out 파일명 규칙 (요청하신대로 "variogram_" 접두사 제거, xlag 값의 '.'을 '_'로 변경)
                    gslib_out_file_base = f"{var_name_for_file}_nlag{nlag_to_use}_xlag{str(xlag_val).replace('.', '_')}"
                    gslib_out_file = f"{gslib_out_file_base}.out"

                    par_file_name_base = f"par_{var_name_for_file}_nlag{nlag_to_use}_xlag{str(xlag_val).replace('.', '_')}"
                    full_par_file_path = os.path.join(
                        OUTPUT_PAR_DIR, f"{par_file_name_base}.par"
                    )

                    variogram_details = [
                        (
                            internal_active_var_idx,
                            internal_active_var_idx,
                            ivtype_for_categorical,
                            cat_value,
                        )
                    ]
                    nvarg_val = 1

                    par_lines = create_gslib_par_content(
                        data_file_for_gslib_par=GSLIB_TARGET_DATA_FILE,
                        x_col_num=X_COL_FOR_GSLIB_PAR,
                        y_col_num=Y_COL_FOR_GSLIB_PAR,
                        z_col_num=Z_COL_FOR_GSLIB_PAR,
                        num_vars_to_activate=1,
                        col_of_var_to_activate=actual_var_column_in_gslib_datafile,
                        trim_min=0.0,
                        trim_max=1.0e21,  # 지시자 변수는 0 또는 1
                        gslib_output_filename=gslib_out_file,
                        nlag=nlag_to_use,
                        xlag=xlag_val,
                        xtol=xtol_val,
                        num_directions=4,
                        direction_lines=[  # GSLIB 기본 방향 설정 예시 (사용자 값 유지)
                            "000.0  45.0  5.0    0.0  22.5  25.0",
                            "045.0  45.0  5.0    0.0  22.5  25.0",
                            "090.0  45.0  5.0    0.0  22.5  25.0",
                            "135.0  45.0  5.0    0.0  22.5  25.0",
                        ],
                        standardize_sill=1,
                        nvarg=nvarg_val,
                        variogram_details_lines=variogram_details,
                    )
                    try:
                        with open(full_par_file_path, "w", encoding="utf-8") as f:
                            for line_entry in par_lines:
                                f.write(line_entry + "\n")
                        par_file_count += 1
                    except Exception as e:
                        print(f"Error writing .par file {full_par_file_path}: {e}")

        elif var_info["type"] == "continuous":
            ivtype_for_continuous = var_info.get("ivtype_cont", 1)
            var_name_for_file = var_name

            for xlag_val in XLAG_VALUES_TO_USE:
                if xlag_val <= 1e-9:
                    print(
                        f"Skipping var='{var_name_for_file}', xlag={xlag_val:.4f} (too small)."
                    )
                    continue

                nlag_to_use = MIN_NLAG_LIMIT
                if max_lag_dist_practical > 1e-9 and xlag_val > 1e-9:
                    nlag_calculated_float = max_lag_dist_practical / xlag_val
                    nlag_to_use = int(round(nlag_calculated_float))
                    if nlag_to_use < MIN_NLAG_LIMIT:
                        nlag_to_use = MIN_NLAG_LIMIT
                    elif nlag_to_use > MAX_NLAG_LIMIT:
                        nlag_to_use = MAX_NLAG_LIMIT
                else:
                    print(
                        f"  Warning: Using default nlag={nlag_to_use} for {var_name_for_file}, xlag={xlag_val:.4f}."
                    )

                xtol_val = xlag_val / 2.0
                if xtol_val <= 1e-9:
                    xtol_val = 1e-9

                gslib_out_file_base = f"{var_name_for_file}_nlag{nlag_to_use}_xlag{str(xlag_val).replace('.', '_')}"
                gslib_out_file = f"{gslib_out_file_base}.out"
                par_file_name_base = f"params_{var_name_for_file}_nlag{nlag_to_use}_xlag{str(xlag_val).replace('.', '_')}"
                full_par_file_path = os.path.join(
                    OUTPUT_PAR_DIR, f"{par_file_name_base}.par"
                )

                variogram_details = [
                    (
                        internal_active_var_idx,
                        internal_active_var_idx,
                        ivtype_for_continuous,
                    )
                ]
                nvarg_val = 1

                par_lines = create_gslib_par_content(
                    data_file_for_gslib_par=GSLIB_TARGET_DATA_FILE,
                    x_col_num=X_COL_FOR_GSLIB_PAR,
                    y_col_num=Y_COL_FOR_GSLIB_PAR,
                    z_col_num=Z_COL_FOR_GSLIB_PAR,
                    num_vars_to_activate=1,
                    col_of_var_to_activate=actual_var_column_in_gslib_datafile,
                    trim_min=0.0,
                    trim_max=1.0e21,  # 연속형 변수의 경우 실제 데이터 범위를 고려하여 설정 가능
                    gslib_output_filename=gslib_out_file,
                    nlag=nlag_to_use,
                    xlag=xlag_val,
                    xtol=xtol_val,
                    num_directions=4,
                    direction_lines=[  # 사용자 제공 값 유지
                        "000.0  45.0  5.0    0.0  22.5  25.0",
                        "045.0  45.0  5.0    0.0  22.5  25.0",
                        "090.0  45.0  5.0    0.0  22.5  25.0",
                        "135.0  45.0  5.0    0.0  22.5  25.0",
                    ],
                    standardize_sill=1,
                    nvarg=nvarg_val,
                    variogram_details_lines=variogram_details,
                )
                try:
                    with open(full_par_file_path, "w", encoding="utf-8") as f:
                        for line_entry in par_lines:
                            f.write(line_entry + "\n")
                    par_file_count += 1
                except Exception as e:
                    print(f"Error writing .par file {full_par_file_path}: {e}")
        else:
            print(
                f"Unknown variable type '{var_info['type']}' for variable {var_name}. Skipping."
            )
            continue

    print(
        f"\nAll {par_file_count} .par files generated in '{os.path.abspath(OUTPUT_PAR_DIR)}'."
    )
    print(
        f"Ensure GSLIB target data file '{GSLIB_TARGET_DATA_FILE}' and coordinate data file '{PYTHON_COORD_DATA_FILE}' are correctly located relative to GSLIB execution path."
    )
    print(
        "Review 'VARIABLE_COLUMN_INDICES_1_BASED' to ensure it matches column order in GSLIB_TARGET_DATA_FILE."
    )
