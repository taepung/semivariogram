import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib  # matplotlib.colormaps 사용 위함 (현재 이 스크립트에서는 직접 사용 안 함)
from sklearn.neighbors import NearestNeighbors  # 최근린 거리 계산용

# from scipy.spatial.distance import pdist # 최대 거리 계산용이었으나, NND 분석만 하므로 제거 가능 (필요 시 유지)
from scipy import stats as scipy_stats  # 왜도, 첨도, Q-Q 플롯용
import dataframe_image as dfi  # 표 이미지 저장용 (pip install dataframe_image)

# GUI 백엔드가 없는 환경에서 실행될 경우를 대비하여 백엔드 명시 (선택적)
try:
    matplotlib.use("Agg")
except Exception as e:
    print(f"Note: Could not set matplotlib backend to 'Agg'. Error: {e}")
import matplotlib.pyplot as plt


def sanitize_filename_component(name_str):
    """파일 이름에 사용할 수 없는 문자를 밑줄로 변경하고, 양끝의 특정 문자 제거"""
    if not name_str:
        return "unknown"
    s_name = re.sub(r"[^\w.-]", "_", str(name_str))
    s_name = re.sub(r'[\\/:*?"<>|]+', "_", s_name)
    s_name = s_name.strip("._ ")
    return s_name if s_name else "unknown"


def load_gslib_dat_file(filepath):
    """
    GSLIB 형식의 .dat 파일을 읽어 Pandas DataFrame으로 반환합니다.
    첫 줄: 제목 (무시)
    둘째 줄: 변수 개수 (n)
    다음 n 줄: 변수 이름
    이후: 데이터
    """
    data_lines = []
    column_names = []
    num_variables = 0
    header_lines_count = 0

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            title_line = f.readline().strip()
            if not title_line and title_line != "":
                raise ValueError(
                    f"File '{filepath}' appears to be empty or does not start with a title line."
                )
            header_lines_count += 1
            try:
                num_vars_line = f.readline().strip()
                if not num_vars_line:
                    raise ValueError("Missing number of variables line.")
                header_lines_count += 1
                num_variables = int(num_vars_line.split()[0])
                if num_variables <= 0:
                    raise ValueError("Number of variables in header must be positive.")
            except Exception as e_nv:
                raise ValueError(
                    f"Could not read or parse number of variables line from '{filepath}' (line {header_lines_count}): {e_nv}"
                )
            for i in range(num_variables):
                col_name_line = f.readline().strip()
                if not col_name_line:
                    raise ValueError(
                        f"Expected {num_variables} column name lines, but found fewer in '{filepath}' at line {header_lines_count + i + 1}."
                    )
                header_lines_count += 1
                column_names.append(sanitize_filename_component(col_name_line))
            for line_idx, line_content in enumerate(f, start=1):
                stripped_line = line_content.strip()
                if stripped_line and not stripped_line.startswith(("#", "!")):
                    try:
                        values = list(map(float, stripped_line.split()))
                        if len(values) == num_variables:
                            data_lines.append(values)
                        else:
                            print(
                                f"Warning: Line {header_lines_count + line_idx} has {len(values)} values, expected {num_variables}. Skipping: '{stripped_line}'"
                            )
                    except ValueError:
                        print(
                            f"Warning: Could not convert data line {header_lines_count + line_idx} to float, skipping: '{stripped_line}'"
                        )
        if not data_lines:
            raise ValueError("No valid numeric data rows found.")
        data_array = np.array(data_lines)
        if len(column_names) == data_array.shape[1]:
            df = pd.DataFrame(data_array, columns=column_names)
        else:
            raise ValueError(
                f"Column name count ({len(column_names)}) does not match data column count ({data_array.shape[1]})."
            )
        print(
            f"Successfully loaded '{filepath}'. Shape: {df.shape}. Columns: {df.columns.tolist()}"
        )
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Data file not found at '{filepath}'")
    except ValueError as ve:
        raise ValueError(f"Error processing GSLIB file structure in '{filepath}': {ve}")
    except Exception as e:
        raise Exception(
            f"An unexpected error occurred while loading data from '{filepath}': {e}"
        )


def plot_spatial_distribution(
    x_coords, y_coords, title="Spatial Distribution", save_path=None, show_plot=True
):
    plt.figure(figsize=(8, 7))
    plt.scatter(
        x_coords, y_coords, s=20, alpha=0.7, edgecolors="k", linewidths=0.5, c="blue"
    )
    plt.xlabel("X-coordinate", fontsize=12)
    plt.ylabel("Y-coordinate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.7)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Spatial distribution plot saved: {save_path}")
    if show_plot:
        plt.show()
    plt.close()


def analyze_basic_statistics(
    data_df, column_info_dict, output_prefix="basic_stats", show_plots=True
):
    if not isinstance(data_df, pd.DataFrame):
        print("Error: Basic statistics analysis requires a Pandas DataFrame input.")
        return None
    stats_summary_list = []
    categorical_summary_dfs = {}
    print(f"\n--- Basic Statistics / Frequency Analysis ---")
    for col_name_in_file, info in column_info_dict.items():
        var_name_display = sanitize_filename_component(
            info.get("display_name", col_name_in_file)
        )
        var_type = info.get("type", "continuous")
        if col_name_in_file not in data_df.columns:
            print(
                f"Warning: Variable '{col_name_in_file}' (as '{var_name_display}') not found in DataFrame columns. Skipping."
            )
            continue
        series_data = data_df[col_name_in_file]
        if series_data.empty:
            print(f"Warning: Series for {var_name_display} is empty. Skipping.")
            continue
        if var_type == "continuous":
            try:
                series_numeric = pd.to_numeric(series_data, errors="coerce").dropna()
                if series_numeric.empty:
                    print(
                        f"Warning: Continuous var '{var_name_display}' is empty after dropna. Skipping."
                    )
                    continue
                desc = series_numeric.describe()
                skewness = scipy_stats.skew(series_numeric)
                kurt = scipy_stats.kurtosis(series_numeric, fisher=True)
                stats_summary_list.append(
                    {
                        "Variable": var_name_display,
                        "Count": int(desc.get("count", 0)),
                        "Mean": desc.get("mean", np.nan),
                        "StdDev": desc.get("std", np.nan),
                        "Min": desc.get("min", np.nan),
                        "25% (Q1)": desc.get("25%", np.nan),
                        "Median (50%)": desc.get("50%", np.nan),
                        "75% (Q3)": desc.get("75%", np.nan),
                        "Max": desc.get("max", np.nan),
                        "Skewness": skewness,
                        "Kurtosis": kurt,
                    }
                )
                plt.figure(figsize=(8, 5))
                plt.hist(
                    series_numeric,
                    bins="auto",
                    edgecolor="k",
                    alpha=0.7,
                    color="skyblue",
                )
                plt.title(f"Histogram of {var_name_display}", fontsize=14)
                plt.xlabel(var_name_display, fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.grid(True, ls=":", alpha=0.6)
                if output_prefix:
                    plt.savefig(
                        f"{output_prefix}_{var_name_display}_hist.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                if show_plots:
                    plt.show()
                plt.close()
                plt.figure(figsize=(6, 6))
                scipy_stats.probplot(series_numeric, dist="norm", plot=plt)
                plt.title(f"Q-Q Plot of {var_name_display}", fontsize=14)
                plt.xlabel("Theoretical Quantiles", fontsize=12)
                plt.ylabel("Sample Quantiles", fontsize=12)
                plt.grid(True, ls=":", alpha=0.6)
                if output_prefix:
                    plt.savefig(
                        f"{output_prefix}_{var_name_display}_qqplot.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                if show_plots:
                    plt.show()
                plt.close()
            except Exception as e_cont:
                print(
                    f"Error processing continuous variable '{var_name_display}': {e_cont}"
                )
        elif var_type == "categorical":
            try:
                series_cat = series_data.fillna("NaN_value").astype(str)
                counts = series_cat.value_counts(dropna=False)
                percentages = (
                    series_cat.value_counts(normalize=True, dropna=False) * 100
                )
                cat_summary_df = pd.DataFrame(
                    {"Frequency": counts, "Percentage (%)": percentages}
                )
                cat_summary_df.index.name = "Category_Value"
                print(f"\nFrequency Table for Categorical Variable: {var_name_display}")
                print(cat_summary_df)
                categorical_summary_dfs[var_name_display] = cat_summary_df
                plt.figure(figsize=(8, 5))
                counts.plot(kind="bar", edgecolor="k", alpha=0.7, color="lightgreen")
                plt.title(
                    f"Frequency of Categories for {var_name_display}", fontsize=14
                )
                plt.xlabel("Category", fontsize=12)
                plt.ylabel("Frequency", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.grid(axis="y", ls=":", alpha=0.6)
                plt.tight_layout()
                if output_prefix:
                    plt.savefig(
                        f"{output_prefix}_{var_name_display}_barplot.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                if show_plots:
                    plt.show()
                plt.close()
            except Exception as e_cat:
                print(
                    f"Error processing categorical variable '{var_name_display}': {e_cat}"
                )
    if stats_summary_list:
        stats_summary_df = pd.DataFrame(stats_summary_list)
        print("\nContinuous Variables Statistics Summary Table:")
        print(stats_summary_df.to_string(index=False))
        if output_prefix:
            stats_summary_df.to_csv(
                f"{output_prefix}_continuous_summary.csv",
                index=False,
                float_format="%.4f",
            )
            print(f"Continuous var stats saved to CSV.")
            try:
                dfi.export(
                    stats_summary_df.style.format(precision=4, na_rep="-").set_caption(
                        "Continuous Variables Statistics"
                    ),
                    f"{output_prefix}_continuous_summary_table.png",
                    table_conversion="matplotlib",
                    dpi=150,
                )
                print(f"Continuous var stats table image saved.")
            except Exception as e_dfi_cont:
                print(f"Could not save continuous stats table as image: {e_dfi_cont}")
    if categorical_summary_dfs and output_prefix:
        for var_n, df_cat_s in categorical_summary_dfs.items():
            df_cat_s.to_csv(
                f"{output_prefix}_{var_n}_cat_summary.csv", float_format="%.2f"
            )
            print(f"Categorical var '{var_n}' summary saved to CSV.")
            try:
                dfi.export(
                    df_cat_s.style.format(
                        {"Percentage (%)": "{:.2f}%"}, na_rep="-"
                    ).set_caption(f"Frequency for {var_n}"),
                    f"{output_prefix}_{var_n}_cat_summary_table.png",
                    table_conversion="matplotlib",
                    dpi=150,
                )
                print(f"Categorical var '{var_n}' table image saved.")
            except Exception as e_dfi_cat:
                print(
                    f"Could not save cat stats table for '{var_n}' as image: {e_dfi_cat}"
                )
    return stats_summary_df if stats_summary_list else pd.DataFrame()


def analyze_and_plot_nearest_neighbor_distances(
    x_coords, y_coords, data_filename_label="Data", output_prefix="nnd", show_plot=True
):
    if x_coords is None or y_coords is None or len(x_coords) < 2 or len(y_coords) < 2:
        print("Error: Not enough data points (<2) for NND analysis.")
        return None, pd.DataFrame()
    points = np.vstack((x_coords, y_coords)).T
    if points.shape[0] < 2:
        print("Error: Less than 2 valid points for NND analysis.")
        return None, pd.DataFrame()
    try:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=-1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        all_nearest_distances = distances[:, 1]
        if len(all_nearest_distances) == 0:
            print("Warning: No nearest distances found.")
            return None, pd.DataFrame()
        ann_mean = np.mean(all_nearest_distances)
        ann_median = np.median(all_nearest_distances)
        ann_min = np.min(all_nearest_distances)
        ann_max = np.max(all_nearest_distances)
        ann_q1 = np.percentile(all_nearest_distances, 25)
        ann_q3 = np.percentile(all_nearest_distances, 75)
        stats_data = {
            "Statistic": [
                "Mean (ANN)",
                "Median",
                "Min",
                "Max",
                "25th Pctl (Q1)",
                "75th Pctl (Q3)",
                "Num Points",
            ],
            "Value": [
                ann_mean,
                ann_median,
                ann_min,
                ann_max,
                ann_q1,
                ann_q3,
                int(len(all_nearest_distances)),
            ],
        }
        nnd_stats_df = pd.DataFrame(stats_data)
        print(f"\n--- NND Analysis for '{data_filename_label}' ---")
        print(nnd_stats_df.to_string(index=False))
        print("-" * 50)
        if output_prefix:
            nnd_csv_path = f"{output_prefix}_summary.csv"
            nnd_table_img_path = f"{output_prefix}_summary_table.png"
            nnd_hist_path = f"{output_prefix}_hist.png"
            nnd_stats_df.to_csv(nnd_csv_path, index=False, float_format="%.4f")
            print(f"NND stats saved to CSV: {nnd_csv_path}")
            try:
                dfi.export(
                    nnd_stats_df.style.format(
                        {"Value": "{:.4f}"}, na_rep="-"
                    ).set_caption(f"NND Stats for {data_filename_label}"),
                    nnd_table_img_path,
                    table_conversion="matplotlib",
                    dpi=150,
                )
                print(f"NND stats table image saved: {nnd_table_img_path}")
            except Exception as e_dfi_nnd:
                print(f"Could not save NND stats table as image: {e_dfi_nnd}")
        plt.figure(figsize=(10, 6))
        plt.hist(
            all_nearest_distances,
            bins="auto",
            edgecolor="k",
            alpha=0.7,
            color="skyblue",
        )
        plt.axvline(
            ann_mean, color="r", ls="--", lw=1.5, label=f"Mean (ANN): {ann_mean:.4f}"
        )
        plt.axvline(
            ann_median, color="g", ls="--", lw=1.5, label=f"Median: {ann_median:.4f}"
        )
        plt.xlabel("NND", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(f"Histogram of NND for '{data_filename_label}'", fontsize=14)
        plt.legend()
        plt.grid(True, ls=":", alpha=0.7)
        if output_prefix:
            plt.savefig(nnd_hist_path, dpi=150, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close()
        return ann_mean, nnd_stats_df
    except Exception as e:
        print(f"Error NND Analysis: {e}")
        return None, pd.DataFrame()


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    # --- 사용자 설정 ---
    DATA_FILENAME = "jura259.dat"
    OUTPUT_FOLDER_NAME = "preliminary_jura_analysis"
    SHOW_PLOTS_IN_SCRIPT = False  # True로 설정하면 스크립트 실행 중 플롯을 화면에 표시

    # GSLIB 파일 내에서 X, Y 좌표 및 분석 대상 변수들의 "이름"
    # 이 이름은 load_gslib_dat_file 함수가 파일에서 읽어온 컬럼명과 일치해야 합니다.
    X_COLUMN_NAME = "Xlocation"
    Y_COLUMN_NAME = "Ylocation"

    # 기초 통계 분석 및 시각화를 수행할 변수 정보 딕셔너리
    # Key: 데이터 파일 내 실제 컬럼명 (대소문자 주의!)
    # Value: {"type": "continuous" 또는 "categorical", "display_name": "보고서/플롯용 이름 (선택적)"}
    VARIABLES_TO_ANALYZE = {
        # GSLIB jura259.dat 표준 컬럼명 예시 (실제 파일과 다를 수 있음)
        "Cd": {"type": "continuous", "display_name": "Cadmium (Cd)"},
        "Cu": {"type": "continuous", "display_name": "Copper (Cu)"},
        "Pb": {"type": "continuous", "display_name": "Lead (Pb)"},
        "Co": {"type": "continuous", "display_name": "Cobalt (Co)"},
        "Cr": {"type": "continuous", "display_name": "Chromium (Cr)"},
        "Ni": {"type": "continuous", "display_name": "Nickel (Ni)"},
        "Zn": {"type": "continuous", "display_name": "Zinc (Zn)"},
        "Rocktype": {"type": "categorical", "display_name": "Rock Type"},
        "Landuse": {"type": "categorical", "display_name": "Land Use"},
    }
    # --- 설정 끝 ---

    current_working_directory = os.getcwd()
    output_save_path_main = os.path.join(current_working_directory, OUTPUT_FOLDER_NAME)
    os.makedirs(output_save_path_main, exist_ok=True)
    print(
        f"Preliminary analysis results will be saved in: {os.path.abspath(output_save_path_main)}"
    )

    base_output_prefix = os.path.join(
        output_save_path_main,
        sanitize_filename_component(os.path.splitext(DATA_FILENAME)[0]),
    )

    print(f"\nAttempting to load data from: {DATA_FILENAME}")
    try:
        data_df = load_gslib_dat_file(DATA_FILENAME)

        x_coords, y_coords = None, None
        if X_COLUMN_NAME in data_df.columns:
            x_coords = data_df[X_COLUMN_NAME].astype(float)
        else:
            raise ValueError(
                f"X-coordinate column '{X_COLUMN_NAME}' not found. Available: {data_df.columns.tolist()}"
            )
        if Y_COLUMN_NAME in data_df.columns:
            y_coords = data_df[Y_COLUMN_NAME].astype(float)
        else:
            raise ValueError(
                f"Y-coordinate column '{Y_COLUMN_NAME}' not found. Available: {data_df.columns.tolist()}"
            )

        if not x_coords.empty and not y_coords.empty:
            print(
                f"\nExtracted {len(x_coords)} X-coordinates and {len(y_coords)} Y-coordinates."
            )
            # 1. 샘플 위치 분포 시각화 및 저장
            plot_spatial_distribution(
                x_coords,
                y_coords,
                title=f"Spatial Distribution from '{DATA_FILENAME}'",
                save_path=f"{base_output_prefix}_sample_distribution.png",
                show_plot=SHOW_PLOTS_IN_SCRIPT,
            )
            # 2. 최근린 거리 분석, 표 및 그림 저장
            avg_nn_dist, _ = analyze_and_plot_nearest_neighbor_distances(
                x_coords,
                y_coords,
                data_filename_label=DATA_FILENAME,
                output_prefix=f"{base_output_prefix}_nnd",  # 저장 파일명 접두사 전달
                show_plot=SHOW_PLOTS_IN_SCRIPT,
            )
            if avg_nn_dist is not None:
                print(
                    f"\nCalculated Average Nearest Neighbor Distance (ANN): {avg_nn_dist:.4f}"
                )
            else:
                print("\nCould not calculate Average Nearest Neighbor Distance.")
        else:
            print(
                f"Error: Could not extract valid X,Y coords. Skipping NND and spatial distribution plot."
            )

        # 3. 주요 변수들에 대한 기초 통계량 분석, 표 및 그림 저장
        if VARIABLES_TO_ANALYZE:
            analysis_column_info = {}
            for var_key_in_file, info_val in VARIABLES_TO_ANALYZE.items():
                if var_key_in_file in data_df.columns:
                    analysis_column_info[var_key_in_file] = info_val
                else:
                    print(
                        f"Warning: Variable '{var_key_in_file}' in VARIABLES_TO_ANALYZE not found in data columns. Skipping."
                    )

            if analysis_column_info:
                analyze_basic_statistics(
                    data_df,
                    column_info_dict=analysis_column_info,
                    output_prefix=f"{base_output_prefix}_var_stats",
                    show_plots=SHOW_PLOTS_IN_SCRIPT,
                )
            else:
                print(
                    "\nWarning: No valid variables found for basic statistics from VARIABLES_TO_ANALYZE."
                )
        else:
            print(
                "\nNo variables specified for basic statistics (VARIABLES_TO_ANALYZE is empty)."
            )

    except FileNotFoundError as fnf_err:
        print(f"CRITICAL ERROR: {fnf_err}")
    except ValueError as val_err:
        print(f"CRITICAL ERROR: Could not process data file. {val_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    print(
        f"\nPreliminary data analysis complete. Results saved in '{os.path.abspath(output_save_path_main)}' directory."
    )
