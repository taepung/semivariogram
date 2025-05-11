import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib  # matplotlib.colormaps 사용 위함
from scipy.optimize import curve_fit
import dataframe_image as dfi  # 표 이미지 저장을 위해 추가

# GUI 백엔드가 없는 환경에서 실행될 경우를 대비하여 백엔드 명시
try:
    matplotlib.use("Agg")
except Exception:
    pass


# --- 전역 헬퍼 함수 ---
def sanitize_name(name_str):
    if not name_str:
        return "unknown"
    s_name = re.sub(r"[^\w.-]+", "_", str(name_str))
    s_name = s_name.strip("_")
    return s_name if s_name else "unknown"


def sanitize_category_as_pure_integer_string(cat_str):
    if not cat_str:
        return None
    match = re.fullmatch(r"(\d+)", str(cat_str).strip())
    if match:
        return match.group(1)
    return None


# --- 이론적 반변량도 모델 함수들 ---
def spherical_model(h, nugget, sill, range_val):
    partial_sill = sill - nugget
    gamma = np.zeros_like(h, dtype=float)
    h_arr = np.asarray(h)
    safe_range_val = range_val if range_val > 1e-9 else 1e-9
    condition1 = (h_arr > 0) & (h_arr <= safe_range_val)
    condition2 = h_arr > safe_range_val
    gamma[condition1] = nugget + partial_sill * (
        1.5 * (h_arr[condition1] / safe_range_val)
        - 0.5 * (h_arr[condition1] / safe_range_val) ** 3
    )
    gamma[condition2] = nugget + partial_sill
    return gamma


def exponential_model(h, nugget, sill, range_val):
    partial_sill = sill - nugget
    h_arr = np.asarray(h)
    safe_range_val = range_val if range_val > 1e-9 else 1e-9
    return nugget + partial_sill * (1 - np.exp(-3 * h_arr / safe_range_val))


def gaussian_model(h, nugget, sill, range_val):
    partial_sill = sill - nugget
    h_arr = np.asarray(h)
    safe_range_val = range_val if range_val > 1e-9 else 1e-9
    return nugget + partial_sill * (1 - np.exp(-3 * (h_arr / safe_range_val) ** 2))


def calculate_rms(experimental_gamma, theoretical_gamma):
    if (
        len(experimental_gamma) == 0
        or len(theoretical_gamma) == 0
        or len(experimental_gamma) != len(theoretical_gamma)
    ):
        return np.nan
    return np.sqrt(
        np.mean((np.asarray(experimental_gamma) - np.asarray(theoretical_gamma)) ** 2)
    )


# --- GSLIB 출력 파싱 함수 ---
def parse_gamv_output_universal(filepath):
    all_grouped_data = {}
    current_group_id = None
    current_direction_id = None
    current_data_lines = []
    header_pattern = re.compile(
        r"^(.*?)\s*tail:(.*?)\s*head:(.*?)\s*direction\s+(\d+)", re.IGNORECASE
    )

    user_specific_vars_theme = sorted(
        [
            "rocktype",
            "landuse",
            "lithology",
            "facies",
            "alteration",
            "mineralization",
            "grade",
            "cd",
            "zn",
            "pb",
            "cu",
            "ni",
            "co",
            "cr",
        ],
        key=len,
        reverse=True,
    )

    # extract_info_from_filename은 parse_gamv_output_universal 내부에서만 사용되므로,
    # 여기에 내부 함수로 정의하거나, 전역 sanitize_name을 사용하도록 함.
    # user_specific_vars_theme은 parse_gamv_output_universal의 스코프에 있으므로 접근 가능.
    def extract_info_from_filename(filepath_str, theme_keywords_list):
        filename_lower = os.path.basename(filepath_str).lower()
        name_part = os.path.splitext(filename_lower)[0]

        extracted_theme = "unknown_file_theme"
        extracted_category_from_fn = None
        extracted_unit_lag = None

        temp_name_part_for_cat_lag = name_part
        for var_key in theme_keywords_list:  # 인자로 받은 리스트 사용
            if re.search(rf"(?:^|[\W_]){re.escape(var_key)}(?:[\W_]|$)", name_part):
                extracted_theme = sanitize_name(var_key)  # 전역 sanitize_name 사용
                try:
                    pattern = rf"{re.escape(var_key)}"
                    temp_name_part_for_cat_lag = re.sub(
                        pattern, "", name_part, count=1, flags=re.IGNORECASE
                    ).strip("_- ")
                except re.error:
                    temp_name_part_for_cat_lag = name_part.replace(
                        var_key, "", 1
                    ).strip("_- ")
                break
        if extracted_theme == "unknown_file_theme":
            extracted_theme = (
                sanitize_name(name_part) if name_part else "unknown_file_theme"
            )
            temp_name_part_for_cat_lag = ""

        search_area_for_cat = (
            temp_name_part_for_cat_lag if temp_name_part_for_cat_lag else name_part
        )
        match_cat_num = re.search(
            r"(?:cat|category|indicator|zone|type|ind)[_ ]?(\d+)", search_area_for_cat
        )
        if match_cat_num and match_cat_num.group(1):
            extracted_category_from_fn = match_cat_num.group(1)
        else:
            match_trailing_num = re.search(r"(?:^|_)(\d+)(?:_|$)", search_area_for_cat)
            if match_trailing_num and match_trailing_num.group(1):
                extracted_category_from_fn = match_trailing_num.group(1)

        if extracted_category_from_fn and not re.fullmatch(
            r"\d+", extracted_category_from_fn
        ):  # 순수 정수만
            extracted_category_from_fn = None

        match_xlag = re.search(r"xlag_?(\d+(?:[._]\d+)?)", name_part)
        if match_xlag and match_xlag.group(1):
            try:
                unit_lag_str = match_xlag.group(1).replace("_", ".")
                extracted_unit_lag = float(unit_lag_str)
            except ValueError:
                extracted_unit_lag = None

        return extracted_theme, extracted_category_from_fn, extracted_unit_lag

    file_theme, file_category, file_unit_lag = extract_info_from_filename(
        filepath, user_specific_vars_theme
    )

    def store_current_block_data():
        nonlocal current_data_lines, current_group_id, current_direction_id, all_grouped_data, file_unit_lag
        if current_group_id and current_direction_id and current_data_lines:
            direction_key = f"direction_{current_direction_id}"
            try:
                df = pd.DataFrame(
                    current_data_lines,
                    columns=[
                        "lag_num",
                        "lag_dist",
                        "gamma",
                        "num_pairs",
                        "tail_mean",
                        "head_mean",
                    ],
                )
                df = df.astype(float)
                df["Unit_Lag_Distance"] = (
                    file_unit_lag if file_unit_lag is not None else np.nan
                )
                if not df.empty:
                    if current_group_id not in all_grouped_data:
                        all_grouped_data[current_group_id] = {}
                    all_grouped_data[current_group_id][direction_key] = df
            except ValueError as ve:
                print(
                    f"  Warning (Store): Could not convert data for {current_group_id}-{direction_key}. Error: {ve}"
                )
            except Exception as e:
                print(
                    f"  Warning (Store): Error processing data for {current_group_id}-{direction_key}. Error: {e}"
                )
        current_data_lines = []

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                header_match = header_pattern.search(line_stripped)
                if header_match:
                    store_current_block_data()
                    raw_keyword_part = header_match.group(1).strip()
                    raw_tail_var = header_match.group(2).strip()
                    new_direction_id = header_match.group(4).strip()
                    s_tail_var = sanitize_name(raw_tail_var)  # 전역 sanitize_name 사용
                    keyword_lower = raw_keyword_part.lower()

                    var_name_for_group = file_theme
                    category_for_group_final = file_category
                    is_indicator_header = "indicator" in keyword_lower

                    if is_indicator_header:
                        if category_for_group_final is None:
                            match_hdr_label = re.search(
                                r"indicator\s+([\w./-]+)", keyword_lower
                            )
                            if match_hdr_label:
                                label_text_from_header = match_hdr_label.group(1)
                                category_for_group_final = (
                                    sanitize_category_as_pure_integer_string(
                                        label_text_from_header
                                    )
                                )  # 전역 함수 사용
                        if var_name_for_group == "unknown_file_theme":
                            for theme_key in user_specific_vars_theme:
                                if theme_key in s_tail_var.lower():
                                    var_name_for_group = theme_key
                                    break
                            if (
                                var_name_for_group == "unknown_file_theme"
                                and s_tail_var.lower() not in ["indicator", "unknown"]
                            ):
                                var_name_for_group = s_tail_var
                    else:
                        category_for_group_final = None
                        if s_tail_var and s_tail_var != "unknown":
                            var_name_for_group = s_tail_var
                        elif var_name_for_group == "unknown_file_theme":
                            first_word = keyword_lower.split(" ", 1)[0]
                            var_name_for_group = (
                                sanitize_name(first_word)
                                if first_word
                                else f"unknown_line{line_num}"
                            )

                    if (
                        var_name_for_group == "unknown_file_theme"
                        and not category_for_group_final
                    ):
                        new_group_id_candidate = f"unknown_var_at_line_{line_num}"
                    elif (
                        var_name_for_group == "unknown_file_theme"
                        and category_for_group_final
                    ):
                        new_group_id_candidate = (
                            f"unknown_theme|{category_for_group_final}"
                        )
                    elif category_for_group_final:
                        new_group_id_candidate = (
                            f"{var_name_for_group}|{category_for_group_final}"
                        )
                    else:
                        new_group_id_candidate = var_name_for_group

                    current_group_id = new_group_id_candidate
                    current_direction_id = new_direction_id
                    current_data_lines = []
                    continue
                if current_group_id and current_direction_id:
                    parts = line_stripped.split()
                    if (
                        parts
                        and parts[0].replace(".", "", 1).replace("-", "", 1).isdigit()
                        and len(parts) >= 6
                    ):
                        try:
                            current_data_lines.append([float(p) for p in parts[:6]])
                        except ValueError:
                            pass
            store_current_block_data()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return {}
    except Exception as e:
        print(f"Error reading or parsing file '{filepath}': {e}")
        return {}
    if not all_grouped_data:
        print(f"Warning: No variogram data parsed from '{os.path.basename(filepath)}'.")
    else:
        print(
            f"DEBUG: Parsed from '{os.path.basename(filepath)}'. Groups: {list(all_grouped_data.keys())}"
        )
    return all_grouped_data


# --- 시각화 및 모델 피팅 함수 ---
def analyze_and_plot_variograms(
    data_for_one_group,
    group_id_label_combined,
    out_filename_base,
    models_to_fit_list,
    min_pairs_for_fit,
    save_dir=".",
    display_plot=False,
):
    if isinstance(group_id_label_combined, str) and "|" in group_id_label_combined:
        variable_name, category_name = group_id_label_combined.split("|", 1)
    else:
        variable_name, category_name = group_id_label_combined, None

    unit_lag_dist_for_this_group = None
    first_direction_key = next(iter(data_for_one_group)) if data_for_one_group else None
    if (
        first_direction_key
        and "Unit_Lag_Distance" in data_for_one_group[first_direction_key].columns
    ):
        unique_lags = data_for_one_group[first_direction_key][
            "Unit_Lag_Distance"
        ].unique()
        valid_lags = [lag for lag in unique_lags if pd.notna(lag)]
        if valid_lags:
            unit_lag_dist_for_this_group = valid_lags[0]
            if len(valid_lags) > 1:
                print(
                    f"  Warning: Multiple unit lag distances found for {variable_name}"
                    f"{f' Cat:{category_name}' if category_name else ''}: {valid_lags}. Using {unit_lag_dist_for_this_group}."
                )

    display_group_id = f"{variable_name}{f' (Cat: {category_name})' if category_name else ''}{f' (UnitLag: {unit_lag_dist_for_this_group:.2f})' if unit_lag_dist_for_this_group is not None else ''}"
    num_directions_found = len(data_for_one_group)
    if num_directions_found == 0:
        print(f"  No direction data for {display_group_id}.")
        return []
    all_fit_results_for_group = []
    fig_summary, axes_summary = plt.subplots(
        nrows=max(1, (num_directions_found + 1) // 2),
        ncols=2 if num_directions_found > 1 else 1,
        figsize=(15, 5 * max(1, (num_directions_found + 1) // 2)),
        squeeze=False,
    )
    axes_summary_flat = axes_summary.flatten()
    plot_idx = 0
    overall_max_h, overall_max_gamma = 0, 0
    for dir_idx, (direction_key, df_experimental) in enumerate(
        data_for_one_group.items()
    ):
        direction_num_str = direction_key.split("_")[-1]
        ax = axes_summary_flat[plot_idx]
        title_for_subplot = f"{display_group_id} - Dir {direction_num_str}"
        if df_experimental.empty or not all(
            c in df_experimental for c in ["lag_dist", "gamma", "num_pairs"]
        ):
            ax.text(
                0.5,
                0.5,
                "No data/cols",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title_for_subplot)
            plot_idx += 1
            continue
        df_valid_fit = df_experimental[
            (df_experimental["num_pairs"] >= min_pairs_for_fit)
            & (df_experimental["lag_dist"] > 1e-9)
        ].copy()
        if df_valid_fit.empty:
            ax.plot(
                df_experimental["lag_dist"],
                df_experimental["gamma"],
                "ko",
                ms=5,
                label="Exp (Not Fitted)",
            )
            ax.text(
                0.5,
                0.5,
                f"Pairs<{min_pairs_for_fit}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="r",
            )
            ax.set_title(title_for_subplot)
            if not df_experimental.empty:
                ax.legend()
            plot_idx += 1
            for model_name in models_to_fit_list:
                all_fit_results_for_group.append(
                    {
                        "Variable": variable_name,
                        "Category": category_name,
                        "Unit_Lag_Distance": unit_lag_dist_for_this_group,
                        "Direction": direction_key,
                        "Model": model_name,
                        "Nugget": np.nan,
                        "Sill": np.nan,
                        "Range": np.nan,
                        "RMS": np.nan,
                        "Fitted": False,
                    }
                )
                continue
        h_exp_for_fit = df_valid_fit["lag_dist"].values
        gamma_exp_for_fit = df_valid_fit["gamma"].values
        sort_indices = np.argsort(h_exp_for_fit)
        h_exp_s = h_exp_for_fit[sort_indices]
        gamma_exp_s = gamma_exp_for_fit[sort_indices]
        if h_exp_s.size > 0:
            overall_max_h = max(overall_max_h, np.max(h_exp_s))
        if gamma_exp_s.size > 0:
            overall_max_gamma = max(overall_max_gamma, np.max(gamma_exp_s))
        ax.plot(h_exp_s, gamma_exp_s, "ko", ms=6, label="Experimental")
        best_model_for_direction, best_rms, best_popt = None, float("inf"), None
        for model_name in models_to_fit_list:
            model_function = {
                "spherical": spherical_model,
                "exponential": exponential_model,
                "gaussian": gaussian_model,
            }.get(model_name)
            if not model_function:
                print(f"  Skip unknown:{model_name}")
                continue
            current_rms, current_popt, fitted_this_model = float("inf"), None, False
            try:
                first_lag_gamma = gamma_exp_s[0] if gamma_exp_s.size > 0 else 0
                min_gamma_val = np.min(gamma_exp_s) if gamma_exp_s.size > 0 else 0
                max_gamma_val = np.max(gamma_exp_s) if gamma_exp_s.size > 0 else 1
                mean_h_val = np.mean(h_exp_s) if h_exp_s.size > 0 else 1
                max_h_val = np.max(h_exp_s) if h_exp_s.size > 0 else 10
                initial_nugget = (
                    first_lag_gamma * 0.5
                    if first_lag_gamma > 1e-9
                    else min_gamma_val * 0.1
                )
                initial_sill = max_gamma_val
                initial_range = mean_h_val * 0.5
                p0 = [initial_nugget, initial_sill, initial_range]
                lb = [0, p0[0] + 1e-6 if p0[0] >= 0 else 1e-6, 1e-3]
                ub = [
                    max_gamma_val * 0.9 if max_gamma_val > 0 else 0.1,
                    max_gamma_val * 1.5,
                    max_h_val * 2.0,
                ]
                if lb[1] <= lb[0]:
                    lb[1] = lb[0] + 1e-6
                if p0[1] <= p0[0]:
                    p0[1] = (
                        p0[0] + (max_gamma_val - p0[0]) * 0.1
                        if max_gamma_val > p0[0]
                        else p0[0] + 1e-5
                    )
                if p0[1] <= p0[0]:
                    p0[1] = p0[0] * 1.1 + 1e-5
                if ub[0] < lb[0]:
                    ub[0] = lb[0] + 1e-5
                popt_temp, pcov_temp = curve_fit(
                    model_function,
                    h_exp_s,
                    gamma_exp_s,
                    p0=p0,
                    bounds=(lb, ub),
                    method="trf",
                    maxfev=10000,
                    ftol=1e-6,
                    xtol=1e-6,
                )
                current_rms = calculate_rms(
                    gamma_exp_s, model_function(h_exp_s, *popt_temp)
                )
                current_popt = popt_temp
                fitted_this_model = True
                if not np.isnan(current_rms) and current_rms < best_rms:
                    best_rms, best_model_for_direction, best_popt = (
                        current_rms,
                        model_name,
                        current_popt,
                    )
            except RuntimeError:
                print(f"  FitFail:{display_group_id}-D{direction_num_str}-{model_name}")
            except Exception as e:
                print(
                    f"  ErrFit:{display_group_id}-D{direction_num_str}-{model_name}:{e}"
                )
            n, s, r = current_popt if current_popt is not None else [np.nan] * 3
            all_fit_results_for_group.append(
                {
                    "Variable": variable_name,
                    "Category": category_name,
                    "Unit_Lag_Distance": unit_lag_dist_for_this_group,
                    "Direction": direction_key,
                    "Model": model_name,
                    "Nugget": n,
                    "Sill": s,
                    "Range": r,
                    "RMS": (
                        current_rms
                        if fitted_this_model and not np.isnan(current_rms)
                        else np.nan
                    ),
                    "Fitted": fitted_this_model,
                }
            )
        if best_model_for_direction and best_popt is not None:
            best_model_func = {
                "spherical": spherical_model,
                "exponential": exponential_model,
                "gaussian": gaussian_model,
            }.get(best_model_for_direction)
            h_th = np.linspace(
                0, np.max(h_exp_s) * 1.1 if h_exp_s.size > 0 else 10, 200
            )
            g_th = best_model_func(h_th, *best_popt)
            rms_txt = (
                f"{best_rms:.4f}"
                if isinstance(best_rms, (int, float))
                and not np.isnan(best_rms)
                and not np.isinf(best_rms)
                else "N/A"
            )
            lbl = f"Best:{best_model_for_direction.capitalize()}\nN:{best_popt[0]:.3f},S:{best_popt[1]:.3f},R:{best_popt[2]:.3f}\nRMS:{rms_txt}"
            ax.plot(h_th, g_th, ls="--", lw=2, label=lbl, color="r")
        ax.set_title(title_for_subplot)
        ax.set_xlabel("Lag Distance(h)")
        ax.set_ylabel(r"$\gamma(h)$")
        ax.legend(fontsize="small")
        ax.grid(True, ls=":", alpha=0.7)
        plot_idx += 1
    for i in range(plot_idx, len(axes_summary_flat)):
        fig_summary.delaxes(axes_summary_flat[i])
    fig_summary.suptitle(
        f"Best Fit Variogram Models: {display_group_id}", fontsize=16, y=1.02
    )
    fig_summary.tight_layout(rect=[0, 0, 1, 0.98])
    if overall_max_gamma > 0:
        for ax_s_idx in range(plot_idx):
            axes_summary_flat[ax_s_idx].set_ylim(
                bottom=min(0, -0.05 * overall_max_gamma), top=overall_max_gamma * 1.15
            )
            if overall_max_h > 0:
                axes_summary_flat[ax_s_idx].set_xlim(left=0, right=overall_max_h * 1.15)
    plot_file = os.path.join(
        save_dir,
        f"{out_filename_base}_{variable_name}{f'_{sanitize_name(category_name)}' if category_name else ''}_summary_fits.png",
    )  # 카테고리명도 sanitize
    try:
        fig_summary.savefig(plot_file, dpi=150)
        print(f"  Summary plot: {plot_file}")
    except Exception as e:
        print(f"  Err saving plot {plot_file}:{e}")
    if display_plot:
        plt.show()
    plt.close(fig_summary)
    return all_fit_results_for_group


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    OUT_FILES_DIRECTORY = "."
    PLOTS_SAVE_SUBDIRECTORY = "variogram_analysis"  # 폴더명 변경
    DISPLAY_PLOTS_ON_SCREEN = False
    MODELS_TO_FIT_LIST = ["spherical", "exponential", "gaussian"]
    MIN_PAIRS_FOR_FIT = 10

    all_fitting_results_across_files = []
    plots_save_path = os.path.join(OUT_FILES_DIRECTORY, PLOTS_SAVE_SUBDIRECTORY)
    os.makedirs(plots_save_path, exist_ok=True)
    print(f"Analysis results will be saved in: {os.path.abspath(plots_save_path)}")

    try:
        out_file_list = sorted(
            [f for f in os.listdir(OUT_FILES_DIRECTORY) if f.lower().endswith(".out")]
        )
    except FileNotFoundError:
        print(f"Err: Dir '{OUT_FILES_DIRECTORY}' not found.")
        out_file_list = []
    if not out_file_list:
        print(f"No .out files found in '{OUT_FILES_DIRECTORY}'.")
    else:
        print(f"Found {len(out_file_list)} .out files to process.")

    for out_filename in out_file_list:
        print(f"\n>>> Processing File: {out_filename} <<<")
        full_file_path = os.path.join(OUT_FILES_DIRECTORY, out_filename)
        parsed_data = parse_gamv_output_universal(full_file_path)
        if parsed_data:
            base_name = os.path.splitext(out_filename)[0]
            for group_id_combined, group_data_for_directions in parsed_data.items():
                print(f"  Analyzing Group ID: {group_id_combined}")
                group_results = analyze_and_plot_variograms(
                    group_data_for_directions,
                    group_id_combined,
                    base_name,
                    MODELS_TO_FIT_LIST,
                    MIN_PAIRS_FOR_FIT,
                    plots_save_path,
                    DISPLAY_PLOTS_ON_SCREEN,
                )
                if group_results:
                    all_fitting_results_across_files.extend(group_results)
        else:
            print(f"  No valid data parsed from '{out_filename}'.")

    if all_fitting_results_across_files:
        results_df = pd.DataFrame(all_fitting_results_across_files)

        results_df["Partial_Sill"] = results_df["Sill"] - results_df["Nugget"]
        results_df["Nugget_to_Sill_Ratio"] = results_df["Nugget"] / results_df["Sill"]
        results_df["Nugget_to_Sill_Ratio"].replace(
            [np.inf, -np.inf], np.nan, inplace=True
        )

        results_df["RMS_sort_key"] = results_df["RMS"].fillna(float("inf"))
        results_df["Category_fillna"] = results_df["Category"].fillna("__NONE__")
        results_df["Unit_Lag_Distance_fillna"] = results_df["Unit_Lag_Distance"].fillna(
            -1
        )

        results_df = results_df.sort_values(
            by=[
                "Variable",
                "Category_fillna",
                "Unit_Lag_Distance_fillna",
                "Direction",
                "RMS_sort_key",
            ]
        )

        results_df["Min_RMS_In_Group"] = (
            results_df[results_df["Fitted"]]
            .groupby(
                ["Variable", "Category_fillna", "Unit_Lag_Distance_fillna", "Direction"]
            )["RMS"]
            .transform("min")
        )
        results_df["Is_Best"] = (
            (results_df["Fitted"])
            & (results_df["RMS"] == results_df["Min_RMS_In_Group"])
            & (results_df["RMS"].notna())
        )

        cols_order = [
            "Variable",
            "Category",
            "Unit_Lag_Distance",
            "Direction",
            "Model",
            "Nugget",
            "Sill",
            "Partial_Sill",
            "Range",
            "RMS",
            "Nugget_to_Sill_Ratio",
            "Fitted",
            "Is_Best",
        ]
        final_cols = [c for c in cols_order if c in results_df.columns]
        results_df_to_save = results_df[final_cols].copy()

        csv_file = os.path.join(
            plots_save_path, "all_variogram_fitting_results_final.csv"
        )
        results_df_to_save.to_csv(csv_file, index=False, float_format="%.4f")
        print(f"\nAll fitting results saved to CSV: {csv_file}")

        excel_file_path = os.path.join(
            plots_save_path, "all_variogram_fitting_results_with_filter.xlsx"
        )
        try:
            df_for_excel = results_df_to_save.copy()
            df_for_excel["Fitted"] = (
                df_for_excel["Fitted"].map({True: "Yes", False: "No"}).fillna("No")
            )
            df_for_excel["Is_Best"] = (
                df_for_excel["Is_Best"].map({True: "Yes", False: "No"}).fillna("No")
            )
            df_for_excel["Category"] = df_for_excel["Category"].fillna("-")
            df_for_excel["Unit_Lag_Distance"] = pd.to_numeric(
                df_for_excel["Unit_Lag_Distance"], errors="coerce"
            ).fillna("-")

            writer = pd.ExcelWriter(excel_file_path, engine="xlsxwriter")
            df_for_excel.to_excel(writer, sheet_name="Variogram_Results", index=False)
            workbook = writer.book
            worksheet = writer.sheets["Variogram_Results"]
            worksheet.autofilter(0, 0, len(df_for_excel), len(df_for_excel.columns) - 1)

            best_fit_format = workbook.add_format({"bg_color": "#FFFF99", "bold": True})
            header_format = workbook.add_format(
                {
                    "bold": True,
                    "text_wrap": True,
                    "valign": "top",
                    "fg_color": "#D7E4BC",
                    "border": 1,
                    "align": "center",
                }
            )

            for col_num, value in enumerate(df_for_excel.columns.values):
                worksheet.write(0, col_num, value, header_format)
            for row_idx, is_best_val in enumerate(df_for_excel["Is_Best"]):
                if is_best_val == "Yes":
                    worksheet.set_row(row_idx + 1, None, best_fit_format)
            for column_idx, column_name in enumerate(df_for_excel.columns):
                max_len = max(
                    df_for_excel[column_name].astype(str).map(len).max(),
                    len(column_name),
                )
                worksheet.set_column(
                    column_idx, column_idx, max_len + 3 if max_len < 25 else 25
                )  # 너비 제한

            writer.close()
            print(f"Excel file with auto-filter saved to: {excel_file_path}")
        except Exception as e_excel:
            print(f"Error saving Excel file with filter: {e_excel}")
            print("Ensure 'xlsxwriter' is installed (pip install xlsxwriter).")

        def highlight_best_rows(row):
            is_best_flag = row.get("Is_Best", False)
            if pd.isna(is_best_flag):
                is_best_flag = False
            return [
                "background-color: #fff799; font-weight: bold;" if is_best_flag else ""
                for _ in row
            ]

        try:
            df_display_img = results_df_to_save.copy()
            df_display_img["Category"] = df_display_img["Category"].fillna("-")
            df_display_img["Unit_Lag_Distance"] = pd.to_numeric(
                df_display_img["Unit_Lag_Distance"], errors="coerce"
            ).fillna("-")

            float_cols_format_img = {
                col: "{:.4f}"
                for col in ["Nugget", "Sill", "Partial_Sill", "Range", "RMS"]
            }
            if "Nugget_to_Sill_Ratio" in df_display_img.columns:
                float_cols_format_img["Nugget_to_Sill_Ratio"] = "{:.2%}"
            if "Unit_Lag_Distance" in df_display_img.columns:
                float_cols_format_img["Unit_Lag_Distance"] = "{:.2f}"

            df_display_img["Fitted"] = (
                df_display_img["Fitted"].map({True: "Y", False: "N"}).fillna("N")
            )
            df_display_img["Is_Best"] = (
                df_display_img["Is_Best"].map({True: "Y", False: "N"}).fillna("N")
            )

            # Variable, Category, Unit_Lag_Distance, Direction, Model, Fitted, Is_Best 는 중앙 정렬
            styled_td_nth_children_img = "td:nth-child(1),td:nth-child(2),td:nth-child(3),td:nth-child(4),td:nth-child(5),td:nth-child(12),td:nth-child(13)"

            df_styled_img = (
                df_display_img.style.apply(highlight_best_rows, axis=1)
                .format(float_cols_format_img, na_rep="-")
                .set_caption(
                    "Variogram Fitting Results (Best fit per direction highlighted)"
                )
                .set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": "text-align:center; font-size:8pt; border:1px solid #bababa; padding:4px; background-color:#f0f0f0;",
                        },
                        {
                            "selector": "td",
                            "props": "text-align:right; font-size:7.5pt; border:1px solid #e0e0e0; padding:3px 4px;",
                        },
                        {
                            "selector": styled_td_nth_children_img,
                            "props": "text-align:center;",
                        },
                    ]
                )
                .set_properties(
                    **{
                        "border-collapse": "collapse",
                        "width": "auto",
                        "margin": "0px auto",
                        "font-family": "Calibri,sans-serif",
                    }
                )
            )

            table_img_file = os.path.join(
                plots_save_path, "all_variogram_fitting_results_table_styled.png"
            )
            try:
                dfi.export(df_styled_img, table_img_file, dpi=200)
                print(
                    f"Styled fitting results table image (dfi) saved to: {table_img_file}"
                )
            except Exception as dfi_e:
                print(f"dfi export failed: {dfi_e}. Trying basic matplotlib table.")
                fig_tb, ax_tb = plt.subplots(
                    figsize=(
                        max(15, len(df_display_img.columns) * 1.2),
                        max(8, len(df_display_img) * 0.3),
                    )
                )
                ax_tb.axis("off")
                ax_tb.axis("tight")
                colWidths_mpl = [
                    0.12 if col not in ["Variable", "Model"] else 0.15
                    for col in df_display_img.columns
                ]
                tb_mpl = ax_tb.table(
                    cellText=df_display_img.round(4).fillna("-").values,
                    colLabels=df_display_img.columns,
                    loc="center",
                    cellLoc="center",
                    colWidths=colWidths_mpl,
                )
                tb_mpl.auto_set_font_size(False)
                tb_mpl.set_fontsize(5)
                tb_mpl.scale(1, 1.0)  # 폰트 크기 더 줄임
                plt.suptitle("Variogram Fitting Results", fontsize=12)
                tb_img_mpl = os.path.join(
                    plots_save_path, "all_variogram_fitting_results_table_mpl.png"
                )
                plt.savefig(tb_img_mpl, dpi=220, bbox_inches="tight")
                plt.close(fig_tb)  # 해상도 약간 높임
                print(f"Fitting results table image (mpl) saved to: {tb_img_mpl}")
        except Exception as e_img:
            print(f"Err table img: {e_img}")

        # --- 변수별, 카테고리별 최적 모델 빈도수 분석 ---
        print("\n\n--- Best Model Frequency Analysis ---")
        df_best_fits_for_freq = results_df_to_save[
            results_df_to_save["Is_Best"] == True
        ].copy()
        if not df_best_fits_for_freq.empty:
            print("\n1. Best Model Frequency per Variable:")
            variable_model_counts = (
                df_best_fits_for_freq.groupby("Variable")["Model"]
                .value_counts()
                .unstack(fill_value=0)
            )
            variable_dominant_model = variable_model_counts.idxmax(axis=1)
            variable_summary = pd.concat(
                [
                    variable_model_counts,
                    variable_dominant_model.rename("Dominant_Model"),
                ],
                axis=1,
            )
            print(variable_summary)
            try:
                ax = variable_model_counts.plot(
                    kind="bar", figsize=(12, 7), rot=45, width=0.8
                )
                plt.title(
                    "Dominant Variogram Model Type per Variable (Based on Best Fits)"
                )
                plt.ylabel("Frequency (Number of Directions)")
                plt.xlabel("Variable")
                plt.tight_layout()
                freq_plot_path = os.path.join(
                    plots_save_path, "variable_model_frequency.png"
                )
                plt.savefig(freq_plot_path, dpi=150)
                print(f"Variable model frequency plot saved to: {freq_plot_path}")
                if DISPLAY_PLOTS_ON_SCREEN:
                    plt.show()
                plt.close()
            except Exception as e_plot_freq:
                print(
                    f"Could not generate variable model frequency plot: {e_plot_freq}"
                )

            if (
                "Category" in df_best_fits_for_freq.columns
                and df_best_fits_for_freq["Category"].notna().any()
                and df_best_fits_for_freq[df_best_fits_for_freq["Category"] != "-"][
                    "Category"
                ].nunique()
                > 0
            ):
                print("\n2. Best Model Frequency per Variable and Category:")
                df_meaningful_cat = df_best_fits_for_freq[
                    df_best_fits_for_freq["Category"].notna()
                    & (df_best_fits_for_freq["Category"] != "-")
                ]
                if not df_meaningful_cat.empty:
                    var_cat_model_counts = (
                        df_meaningful_cat.groupby(["Variable", "Category"])["Model"]
                        .value_counts()
                        .unstack(fill_value=0)
                    )
                    var_cat_dominant_model = var_cat_model_counts.idxmax(axis=1)
                    var_cat_summary = pd.concat(
                        [
                            var_cat_model_counts,
                            var_cat_dominant_model.rename("Dominant_Model"),
                        ],
                        axis=1,
                    )
                    print(var_cat_summary)
                    try:
                        for var_name, group_data in var_cat_model_counts.groupby(
                            level="Variable"
                        ):
                            ax_cat = group_data.droplevel("Variable").plot(
                                kind="bar", figsize=(10, 6), rot=30, width=0.7
                            )
                            plt.title(f"Dominant Model for {var_name} by Category")
                            plt.ylabel("Frequency")
                            plt.xlabel("Category")
                            plt.tight_layout()
                            cat_freq_plot_path = os.path.join(
                                plots_save_path,
                                f"{sanitize_name(var_name)}_category_model_frequency.png",
                            )
                            plt.savefig(cat_freq_plot_path, dpi=150)
                            print(
                                f"Category model frequency plot for {var_name} saved to: {cat_freq_plot_path}"
                            )
                            if DISPLAY_PLOTS_ON_SCREEN:
                                plt.show()
                            plt.close()
                    except Exception as e_plot_cat_freq:
                        print(
                            f"Could not generate category model frequency plots: {e_plot_cat_freq}"
                        )
                else:
                    print(
                        "  No data with meaningful categories for frequency analysis."
                    )
            else:
                print(
                    "  Category column not present or no meaningful category data for frequency analysis."
                )
        else:
            print("  No best fit models found to analyze frequency.")

    print("\nAll processing complete.")
    if out_file_list and os.path.exists(plots_save_path):
        print(f"Results in: {os.path.abspath(plots_save_path)}")
