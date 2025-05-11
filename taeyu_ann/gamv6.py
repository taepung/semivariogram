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
    if partial_sill < 0:
        partial_sill = 0
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
    if partial_sill < 0:
        partial_sill = 0
    h_arr = np.asarray(h)
    safe_range_val = range_val if range_val > 1e-9 else 1e-9
    return nugget + partial_sill * (1 - np.exp(-3 * h_arr / safe_range_val))


def gaussian_model(h, nugget, sill, range_val):
    partial_sill = sill - nugget
    if partial_sill < 0:
        partial_sill = 0
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

    def extract_info_from_filename(filepath_str, theme_keywords_list):
        filename_lower = os.path.basename(filepath_str).lower()
        name_part = os.path.splitext(filename_lower)[0]
        extracted_theme = "unknown_file_theme"
        extracted_category_from_fn = None
        extracted_unit_lag = None
        remaining_part_for_cat_lag = name_part
        for var_key in theme_keywords_list:
            if re.search(rf"(?:^|[\W_]){re.escape(var_key)}(?:[\W_]|$)", name_part):
                extracted_theme = sanitize_name(var_key)
                try:
                    pattern = rf"{re.escape(var_key)}"
                    remaining_part_for_cat_lag = re.sub(
                        pattern, "", name_part, count=1, flags=re.IGNORECASE
                    ).strip("_- ")
                except re.error:
                    remaining_part_for_cat_lag = name_part.replace(
                        var_key, "", 1
                    ).strip("_- ")
                break
        if extracted_theme == "unknown_file_theme" and name_part:
            extracted_theme = sanitize_name(name_part)
            remaining_part_for_cat_lag = ""
        search_area_for_cat = (
            remaining_part_for_cat_lag if remaining_part_for_cat_lag else name_part
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
        ):
            extracted_category_from_fn = None
        match_xlag = re.search(r"xlag_?(\d+(?:[._]\d+)?)", name_part)
        if match_xlag and match_xlag.group(1):
            try:
                extracted_unit_lag = float(match_xlag.group(1).replace("_", "."))
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
                    s_tail_var = sanitize_name(raw_tail_var)
                    keyword_lower = raw_keyword_part.lower()
                    var_name_for_group = file_theme
                    category_for_group_final = file_category
                    is_indicator_header = "indicator" in keyword_lower
                    if is_indicator_header:
                        if category_for_group_final is None:
                            match_hdr_label = re.search(
                                r"indicator\s+(\d+)", keyword_lower
                            )
                            if match_hdr_label:
                                category_for_group_final = match_hdr_label.group(1)
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
    representative_model_type,
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
                f"  Warning: Multiple unit lag distances for {variable_name}{f' Cat:{category_name}' if category_name else ''}: {valid_lags}. Using {unit_lag_dist_for_this_group}."
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
    model_function_to_use = {
        "spherical": spherical_model,
        "exponential": exponential_model,
        "gaussian": gaussian_model,
    }.get(representative_model_type)
    if not model_function_to_use:
        print(
            f"  Error: Rep. model '{representative_model_type}' undefined for {display_group_id}. Plotting exp. data only."
        )
        for dir_idx, (direction_key, df_experimental) in enumerate(
            data_for_one_group.items()
        ):
            ax = axes_summary_flat[plot_idx]
            title_for_subplot = (
                f"{display_group_id} - Dir {direction_key.split('_')[-1]}"
            )
            if not df_experimental.empty:
                ax.plot(
                    df_experimental["lag_dist"],
                    df_experimental["gamma"],
                    "ko",
                    ms=5,
                    label="Experimental Data",
                )
            ax.set_title(title_for_subplot)
            ax.legend(fontsize="small")
            ax.grid(True, ls=":", alpha=0.7)
            plot_idx += 1
            all_fit_results_for_group.append(
                {
                    "Variable": variable_name,
                    "Category": category_name,
                    "Unit_Lag_Distance": unit_lag_dist_for_this_group,
                    "Direction": direction_key,
                    "Model": representative_model_type,
                    "Nugget": np.nan,
                    "Sill": np.nan,
                    "Range": np.nan,
                    "RMS": np.nan,
                    "Fitted": False,
                }
            )
        for i in range(plot_idx, len(axes_summary_flat)):
            fig_summary.delaxes(axes_summary_flat[i])
        fig_summary.suptitle(
            f"Experimental Variograms (No fitting model '{representative_model_type}' defined): {display_group_id}",
            fontsize=14,
            y=0.99,
        )
        fig_summary.tight_layout(rect=[0, 0.03, 1, 0.95])  # 제목 공간 확보
        plot_file = os.path.join(
            save_dir,
            f"{out_filename_base}_{sanitize_name(variable_name)}{f'_{sanitize_name(category_name)}' if category_name else ''}_exp_only.png",
        )
        try:
            fig_summary.savefig(plot_file, dpi=150, bbox_inches="tight")
            print(f"  Experimental plot saved: {plot_file}")
        except Exception as e:
            print(f"  Err saving exp plot {plot_file}:{e}")
        if display_plot:
            plt.show()
        plt.close(fig_summary)
        return all_fit_results_for_group

    for dir_idx, (direction_key, df_experimental) in enumerate(
        data_for_one_group.items()
    ):
        direction_num_str = direction_key.split("_")[-1]
        ax = axes_summary_flat[plot_idx]
        title_for_subplot = f"Dir {direction_num_str}"
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
                ax.legend(fontsize="small")
            plot_idx += 1
            all_fit_results_for_group.append(
                {
                    "Variable": variable_name,
                    "Category": category_name,
                    "Unit_Lag_Distance": unit_lag_dist_for_this_group,
                    "Direction": direction_key,
                    "Model": representative_model_type,
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
        current_rms, current_popt, fitted_this_model = float("inf"), None, False
        try:
            first_lag_gamma = gamma_exp_s[0] if gamma_exp_s.size > 0 else 0
            min_gamma_val = np.min(gamma_exp_s) if gamma_exp_s.size > 0 else 0
            max_gamma_val = np.max(gamma_exp_s) if gamma_exp_s.size > 0 else 1
            mean_h_val = np.mean(h_exp_s) if h_exp_s.size > 0 else 1
            max_h_val = np.max(h_exp_s) if h_exp_s.size > 0 else 10
            initial_nugget = (
                first_lag_gamma * 0.5 if first_lag_gamma > 1e-9 else min_gamma_val * 0.1
            )
            initial_sill = max_gamma_val
            initial_range = mean_h_val * 0.5
            p0 = [initial_nugget, initial_sill, initial_range]
            lb = [0, p0[0] + 1e-6 if p0[0] >= 0 else 1e-6, 1e-3]
            ub = [
                max_gamma_val * 0.95 if max_gamma_val > 0 else 0.1,
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
                model_function_to_use,
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
                gamma_exp_s, model_function_to_use(h_exp_s, *popt_temp)
            )
            current_popt = popt_temp
            fitted_this_model = True
        except RuntimeError:
            print(
                f"  FitFail:{display_group_id}-D{direction_num_str}-{representative_model_type}"
            )
        except Exception as e:
            print(
                f"  ErrFit:{display_group_id}-D{direction_num_str}-{representative_model_type}:{e}"
            )
        n, s, r = current_popt if current_popt is not None else [np.nan] * 3
        all_fit_results_for_group.append(
            {
                "Variable": variable_name,
                "Category": category_name,
                "Unit_Lag_Distance": unit_lag_dist_for_this_group,
                "Direction": direction_key,
                "Model": representative_model_type,
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
        if fitted_this_model and current_popt is not None:
            h_th = np.linspace(
                0, np.max(h_exp_s) * 1.1 if h_exp_s.size > 0 else 10, 200
            )
            g_th = model_function_to_use(h_th, *current_popt)
            rms_txt = (
                f"{current_rms:.4f}"
                if isinstance(current_rms, (int, float))
                and not np.isnan(current_rms)
                and not np.isinf(current_rms)
                else "N/A"
            )
            lbl = f"Fit:{representative_model_type.capitalize()}\nN:{current_popt[0]:.3f},S:{current_popt[1]:.3f},R:{current_popt[2]:.3f}\nRMS:{rms_txt}"
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
        f"Variogram Fitting: {display_group_id}", fontsize=14, y=0.99
    )  # y값 조정
    fig_summary.tight_layout(rect=[0, 0.03, 1, 0.95])  # 상단 여백 확보
    if overall_max_gamma > 0:
        for ax_s_idx in range(plot_idx):
            axes_summary_flat[ax_s_idx].set_ylim(
                bottom=min(0, -0.05 * overall_max_gamma), top=overall_max_gamma * 1.15
            )
            if overall_max_h > 0:
                axes_summary_flat[ax_s_idx].set_xlim(left=0, right=overall_max_h * 1.15)
    plot_file_name_suffix = f"{representative_model_type}_fits"
    plot_file = os.path.join(
        save_dir,
        f"{out_filename_base}_{sanitize_name(variable_name)}{f'_{sanitize_name(category_name)}' if category_name else ''}_{plot_file_name_suffix}.png",
    )
    try:
        fig_summary.savefig(plot_file, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {plot_file}")  # bbox_inches 추가
    except Exception as e:
        print(f"  Err saving plot {plot_file}:{e}")
    if display_plot:
        plt.show()
    plt.close(fig_summary)
    return all_fit_results_for_group


# --- 메인 실행 블록 ---
if __name__ == "__main__":
    OUT_FILES_DIRECTORY = "."
    PLOTS_SAVE_SUBDIRECTORY = "variogram_analysis_rep_model"
    DISPLAY_PLOTS_ON_SCREEN = False
    MIN_PAIRS_FOR_FIT = 10

    REPRESENTATIVE_MODELS_CONFIG = {
        "Cd": "spherical",
        "Co": "spherical",
        "Cr": "spherical",
        "Cu": "gaussian",
        "Ni": "spherical",
        "Pb": "spherical",
        "Zn": "exponential",
        ("landuse", "1"): "spherical",
        ("landuse", "2"): "gaussian",
        ("landuse", "3"): "gaussian",
        ("landuse", "4"): "gaussian",
        ("rocktype", "1"): "exponential",
        ("rocktype", "2"): "exponential",
        ("rocktype", "3"): "gaussian",
        ("rocktype", "4"): "gaussian",
        ("rocktype", "5"): "gaussian",
    }
    DEFAULT_MODEL_IF_NOT_SPECIFIED = "spherical"

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
                current_var, current_cat = (
                    (group_id_combined.split("|", 1) + [None])[:2]
                    if "|" in group_id_combined
                    else (group_id_combined, None)
                )
                model_key = (current_var, current_cat) if current_cat else current_var
                representative_model = REPRESENTATIVE_MODELS_CONFIG.get(model_key)
                if representative_model is None and isinstance(
                    model_key, tuple
                ):  # (var,cat) 키가 없으면 var 단독키로 다시 시도
                    representative_model = REPRESENTATIVE_MODELS_CONFIG.get(
                        current_var, DEFAULT_MODEL_IF_NOT_SPECIFIED
                    )
                elif representative_model is None:  # var 단독키도 없으면 기본값 사용
                    representative_model = DEFAULT_MODEL_IF_NOT_SPECIFIED
                print(f"    Using representative model: {representative_model}")
                group_results = analyze_and_plot_variograms(
                    group_data_for_directions,
                    group_id_combined,
                    base_name,
                    representative_model_type=representative_model,
                    min_pairs_for_fit=MIN_PAIRS_FOR_FIT,
                    save_dir=plots_save_path,
                    display_plot=DISPLAY_PLOTS_ON_SCREEN,
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
        results_df["Is_Best"] = results_df[
            "Fitted"
        ]  # 대표모델 하나만 사용하므로 Fitted=Is_Best
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
            plots_save_path, "representative_model_fitting_results.csv"
        )
        results_df_to_save.to_csv(csv_file, index=False, float_format="%.4f")
        print(f"\nRepresentative model fitting results saved to CSV: {csv_file}")

        excel_file_path = os.path.join(
            plots_save_path, "representative_model_fitting_results_with_filter.xlsx"
        )
        try:
            df_for_excel = results_df_to_save.copy()
            df_for_excel["Fitted_Display"] = (
                df_for_excel["Fitted"].map({True: "Yes", False: "No"}).fillna("No")
            )
            df_for_excel["Is_Best_Display"] = df_for_excel["Fitted_Display"]
            df_for_excel["Category"] = df_for_excel["Category"].fillna("-")
            df_for_excel["Unit_Lag_Distance"] = pd.to_numeric(
                df_for_excel["Unit_Lag_Distance"], errors="coerce"
            ).fillna("-")
            excel_cols = [
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
                "Fitted_Display",
                "Is_Best_Display",
            ]
            excel_cols_final = [
                c
                for c in excel_cols
                if c.replace("_Display", "") in df_for_excel.columns
                or c in df_for_excel.columns
            ]
            writer = pd.ExcelWriter(excel_file_path, engine="xlsxwriter")
            df_for_excel[excel_cols_final].to_excel(
                writer, sheet_name="Rep_Model_Results", index=False
            )
            workbook = writer.book
            worksheet = writer.sheets["Rep_Model_Results"]
            worksheet.autofilter(0, 0, len(df_for_excel), len(excel_cols_final) - 1)
            fitted_highlight_format = workbook.add_format(
                {"bg_color": "#E6FFCC", "bold": False}
            )
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
            for col_num, value in enumerate(
                df_for_excel[excel_cols_final].columns.values
            ):
                worksheet.write(0, col_num, value, header_format)
            for row_idx, fitted_val in enumerate(df_for_excel["Fitted_Display"]):
                if fitted_val == "Yes":
                    worksheet.set_row(row_idx + 1, None, fitted_highlight_format)
            for col_idx, col_name in enumerate(df_for_excel[excel_cols_final].columns):
                max_len = max(
                    df_for_excel[col_name].astype(str).map(len).max(), len(col_name)
                )
                worksheet.set_column(
                    col_idx, col_idx, max_len + 3 if max_len < 25 else 25
                )
            writer.close()
            print(
                f"Excel file with auto-filter (Representative Model) saved to: {excel_file_path}"
            )
        except Exception as e_excel:
            print(f"Error saving Rep. Model Excel: {e_excel}")

        def highlight_fitted_rows(row):
            is_fitted_flag = row.get("Fitted", False)
            if pd.isna(is_fitted_flag):
                is_fitted_flag = False
            return [
                (
                    "background-color: #E6FFCC; font-weight: normal;"
                    if is_fitted_flag
                    else ""
                )
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
            df_display_img["Fitted_Display"] = (
                df_display_img["Fitted"].map({True: "Y", False: "N"}).fillna("N")
            )
            df_display_img["Is_Best_Display"] = df_display_img["Fitted_Display"]

            # 테이블 이미지용 컬럼 (Fitted, Is_Best 원본 bool 대신 Yes/No 사용)
            df_img_cols = [
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
                "Fitted_Display",
                "Is_Best_Display",
            ]
            df_img_final_cols = [
                c
                for c in df_img_cols
                if c.replace("_Display", "") in df_display_img.columns
                or c in df_display_img.columns
            ]
            df_to_style = df_display_img[df_img_final_cols].rename(
                columns={"Fitted_Display": "Fitted", "Is_Best_Display": "Is_Best"}
            )

            styled_td_nth_children_img = "td:nth-child(1),td:nth-child(2),td:nth-child(3),td:nth-child(4),td:nth-child(5),td:nth-child(12),td:nth-child(13)"
            df_styled_img = (
                df_to_style.style.apply(highlight_fitted_rows, axis=1)
                .format(float_cols_format_img, na_rep="-")
                .set_caption("Variogram Fitting Results (Fitted models highlighted)")
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
                plots_save_path, "representative_model_fitting_table_styled.png"
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
                        max(15, len(df_to_style.columns) * 1.2),
                        max(8, len(df_to_style) * 0.3),
                    )
                )
                ax_tb.axis("off")
                ax_tb.axis("tight")
                colWidths_mpl = [
                    0.12 if col not in ["Variable", "Model"] else 0.15
                    for col in df_to_style.columns
                ]
                tb_mpl = ax_tb.table(
                    cellText=df_to_style.round(4).fillna("-").values,
                    colLabels=df_to_style.columns,
                    loc="center",
                    cellLoc="center",
                    colWidths=colWidths_mpl,
                )
                tb_mpl.auto_set_font_size(False)
                tb_mpl.set_fontsize(5)
                tb_mpl.scale(1, 1.0)
                plt.suptitle("Variogram Fitting Results", fontsize=12)
                tb_img_mpl = os.path.join(
                    plots_save_path, "representative_model_fitting_table_mpl.png"
                )
                plt.savefig(tb_img_mpl, dpi=220, bbox_inches="tight")
                plt.close(fig_tb)
                print(f"Fitting results table image (mpl) saved to: {tb_img_mpl}")
        except Exception as e_img:
            print(f"Err table img: {e_img}")

        # 대표 모델 사용 시 빈도수 분석은 각 (Var,Cat,UnitLag,Dir) 조합당 하나의 모델만 있으므로,
        # Fitted == True 인 경우의 카운트로 변경
        print(
            "\n\n--- Representative Model Fit Counts (Number of Fitted Directions) ---"
        )
        fitted_counts = (
            results_df_to_save[results_df_to_save["Fitted"] == True]
            .groupby(["Variable", "Category", "Unit_Lag_Distance", "Model"])[
                "Direction"
            ]
            .nunique()
            .unstack(fill_value=0)
        )  # Model별로 몇개의 Direction이 성공했는지
        if not fitted_counts.empty:
            print(fitted_counts)
            try:
                ax = fitted_counts.plot(
                    kind="bar", stacked=False, figsize=(12, 7), rot=45, width=0.8
                )  # stacked=False로 변경
                plt.title(
                    "Number of Successfully Fitted Directions per Group and Model"
                )
                plt.ylabel("Number of Directions Fitted")
                plt.tight_layout()
                fit_count_plot_path = os.path.join(
                    plots_save_path, "representative_model_fit_counts.png"
                )
                plt.savefig(fit_count_plot_path, dpi=150)
                print(f"Fit count plot saved to: {fit_count_plot_path}")
                if DISPLAY_PLOTS_ON_SCREEN:
                    plt.show()
                plt.close()
            except Exception as e_fc_plot:
                print(f"Could not generate fit count plot: {e_fc_plot}")
        else:
            print("  No successfully fitted models to summarize for fit counts.")

    print("\nAll processing complete.")
    if out_file_list and os.path.exists(plots_save_path):
        print(f"Results in: {os.path.abspath(plots_save_path)}")
