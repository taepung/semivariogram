import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib  # matplotlib.colormaps 사용 위함

# GUI 백엔드가 없는 환경에서 실행될 경우를 대비하여 백엔드 명시
try:
    matplotlib.use("Agg")
except Exception:
    pass


def parse_gamv_output_universal(filepath):  # <<<--- 이 함수 정의가 정확히 있는지 확인
    """
    GSLIB GAMV .out 파일을 파싱합니다.
    지시자 베리오그램(파일 내 여러 Indicator 컷오프 포함)과 일반 세미베리오그램 등을 모두 처리 시도합니다.
    """
    all_grouped_data = {}
    current_group_id = None
    current_direction_id = None
    current_data_lines = []

    # Updated header pattern to be more general and capture tail/head variable names better.
    # It captures: 1:full_keyword_string, 2:tail_variable, 3:head_variable, 4:direction_number
    header_pattern = re.compile(
        r"^(.*?)\s*tail:(.*?)\s*head:(.*?)\s*direction\s+(\d+)", re.IGNORECASE
    )

    def sanitize_name(name):
        """Helper function to create a safe string for keys/filenames."""
        if not name:
            return "unknown"
        # Replace sequences of non-alphanumeric characters (excluding '.', '-') with a single underscore.
        s_name = re.sub(r"[^\w.-]+", "_", str(name))
        # Remove leading/trailing underscores that might result from replacement.
        s_name = s_name.strip("_")
        # If the name becomes empty after sanitization (e.g., was just "!!!"), return "unknown".
        return s_name if s_name else "unknown"

    def store_current_block_data():
        nonlocal current_data_lines, current_group_id, current_direction_id, all_grouped_data
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
                if not df.empty:
                    if current_group_id not in all_grouped_data:
                        all_grouped_data[current_group_id] = {}
                    all_grouped_data[current_group_id][direction_key] = df
            except ValueError as ve:
                print(
                    f"  Warning (Store): Could not convert/store data for {current_group_id}-{direction_key} in '{os.path.basename(filepath)}'. Error: {ve}"
                )
            except Exception as e:
                print(
                    f"  Warning (Store): Error processing data for {current_group_id}-{direction_key} in '{os.path.basename(filepath)}'. Error: {e}"
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
                    raw_head_var = header_match.group(3).strip()
                    new_direction_id = header_match.group(4).strip()

                    s_tail_var = sanitize_name(raw_tail_var)
                    s_head_var = sanitize_name(raw_head_var)

                    new_group_id_candidate = (
                        f"unknown_{s_tail_var}_line{line_num}"  # Default fallback
                    )
                    keyword_lower = raw_keyword_part.lower()

                    # 1. Indicator variograms
                    #    Regex matches "indicator" followed by optional space, optional '(', a label, optional ')'
                    #    Label can contain \w (alphanumeric + underscore), '.', '/', '-'
                    #    e.g., "indicator 1/2", "indicator (0.25)", "indicator cutoff-A"
                    indicator_keyword_match = re.match(
                        r"indicator\s*\(?([\w./-]+)\)?", keyword_lower
                    )

                    if indicator_keyword_match:
                        indicator_label_text = indicator_keyword_match.group(1)
                        s_indicator_label = sanitize_name(indicator_label_text)
                        new_group_id_candidate = f"indicator_{s_indicator_label}"
                    elif (
                        "indicator" in keyword_lower
                    ):  # Generic indicator keyword without a clear label in keyword part
                        if s_tail_var and s_tail_var not in [
                            "indicator",
                            "unknown",
                        ]:  # Use tail variable if descriptive
                            new_group_id_candidate = f"indicator_var_{s_tail_var}"
                        else:  # Fallback for very generic indicators
                            new_group_id_candidate = f"indicator_generic_line{line_num}"
                    # 2. Cross-variograms (tail != head)
                    elif s_tail_var != s_head_var:
                        cross_base_type = "cross_variogram"
                        if "covariance" in keyword_lower:
                            cross_base_type = "cross_covariance"
                        elif "correlogram" in keyword_lower:
                            cross_base_type = "cross_correlogram"
                        # Note: "semivariogram" or "variogram" in keyword_lower implies cross_variogram
                        new_group_id_candidate = (
                            f"{cross_base_type}_{s_tail_var}_vs_{s_head_var}"
                        )
                    # 3. Direct variograms (tail == head) or other types
                    else:
                        direct_base_type = "variogram"  # Default
                        if "semivariogram" in keyword_lower:
                            direct_base_type = "semivariogram"
                        elif "covariance" in keyword_lower:
                            direct_base_type = "covariance"
                        elif "correlogram" in keyword_lower:
                            direct_base_type = "correlogram"
                        elif (
                            "variogram" in keyword_lower
                        ):  # Catches general "variogram"
                            direct_base_type = "variogram"
                        else:  # Fallback: try to use the first word of the raw_keyword_part (sanitized)
                            first_word_raw = keyword_lower.split(" ", 1)[0]
                            if first_word_raw:
                                direct_base_type = sanitize_name(first_word_raw)

                        if s_tail_var and s_tail_var != "unknown":
                            new_group_id_candidate = f"{direct_base_type}_{s_tail_var}"
                        else:  # If tail var is not descriptive, use base_type + line number
                            new_group_id_candidate = (
                                f"{direct_base_type}_line{line_num}"
                            )

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
                            # Silently skip lines that look like data but aren't fully convertible
                            pass

            store_current_block_data()  # Store the last block of data

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return {}
    except Exception as e:
        print(f"Error reading or parsing file '{filepath}': {e}")
        return {}

    if not all_grouped_data:
        print(
            f"Warning: No variogram data could be parsed from '{os.path.basename(filepath)}'. Final top-level keys: {list(all_grouped_data.keys())}"
        )
    else:
        print(
            f"DEBUG: Parsed successfully from '{os.path.basename(filepath)}'. Top-level groups found: {list(all_grouped_data.keys())}"
        )

    return all_grouped_data


def plot_experimental_variograms_universal(
    data_for_one_group,
    group_id_label,
    out_filename_base,
    save_dir=".",
    display_plot=False,
):

    # Note: This plotting function's title/label generation might need adjustments
    # based on the new group_id_label formats (e.g., "semivariogram_Cu", "indicator_1_2")
    # For example, how it extracts 'indicator_category_value' or 'var_name_part'.
    num_directions_found = len(data_for_one_group)
    if num_directions_found == 0:
        print(
            f"  No direction data to plot for {group_id_label} in '{out_filename_base}'."
        )
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    if num_directions_found <= 10:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    elif num_directions_found <= 20:
        cmap = matplotlib.colormaps.get_cmap("tab20")
    else:
        cmap = matplotlib.colormaps.get_cmap("viridis")

    colors_to_use = [
        cmap(i)
        for i in np.linspace(
            0, 0.9, num_directions_found if num_directions_found > 0 else 1
        )
    ]

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "+", "x"]
    max_gamma_overall = 0.0
    plotted_something = False

    for i, (direction_key, df) in enumerate(data_for_one_group.items()):
        if df.empty or not all(
            col in df.columns for col in ["lag_dist", "gamma", "num_pairs"]
        ):
            print(
                f"  Warning: DataFrame for {direction_key} of {group_id_label} in '{out_filename_base}' is empty/missing columns."
            )
            continue

        direction_num_str = direction_key.split("_")[-1]
        valid_df = df[df["num_pairs"] > 0].copy()

        if valid_df.empty:
            print(
                f"  Warning: No valid data pairs for {direction_key} of {group_id_label} in '{out_filename_base}'."
            )
            continue

        current_color = colors_to_use[i % len(colors_to_use)]

        ax.plot(
            valid_df["lag_dist"],
            valid_df["gamma"],
            marker=markers[i % len(markers)],
            color=current_color,
            linestyle="-",
            linewidth=1.2,
            markersize=6,
            label=f"Dir {direction_num_str} (Pairs: {int(valid_df['num_pairs'].min())}-{int(valid_df['num_pairs'].max())})",
        )
        plotted_something = True
        if not valid_df["gamma"].empty:
            current_max_gamma_for_direction = valid_df["gamma"].max()
            if current_max_gamma_for_direction > max_gamma_overall:
                max_gamma_overall = current_max_gamma_for_direction

    if not plotted_something:
        print(
            f"  No data was actually plotted for {group_id_label} in '{out_filename_base}'."
        )
        plt.close(fig)
        return

    # --- Adjust Title and Y-axis label generation based on new group_id_label format ---
    y_axis_label = "Semivariogram $\gamma(h)$"  # Default
    title_group_part = group_id_label.replace("_", " ").title()  # Generic title part

    gid_lower = group_id_label.lower()

    if gid_lower.startswith("indicator_"):
        # Extracts "0.25" from "indicator_0.25" or "1_2" from "indicator_1_2"
        indicator_val_str = group_id_label[len("indicator_") :].replace(
            "_", "/"
        )  # Try to revert common sanitization
        y_axis_label = f"Indicator Semivariogram $\gamma_I(h; cut={indicator_val_str})$"
        title_group_part = f"Indicator for Cutoff {indicator_val_str}"
    elif gid_lower.startswith("semivariogram_"):
        var_name = group_id_label[len("semivariogram_") :].replace("_", " ").title()
        y_axis_label = f"Semivariogram for {var_name} $\gamma(h)$"
        title_group_part = f"Semivariogram for {var_name}"
    elif gid_lower.startswith("covariance_"):
        var_name = group_id_label[len("covariance_") :].replace("_", " ").title()
        y_axis_label = f"Covariance for {var_name} $C(h)$"  # Example, adjust symbol
        title_group_part = f"Covariance for {var_name}"
    elif gid_lower.startswith("correlogram_"):
        var_name = group_id_label[len("correlogram_") :].replace("_", " ").title()
        y_axis_label = f"Correlogram for {var_name} $\rho(h)$"  # Example, adjust symbol
        title_group_part = f"Correlogram for {var_name}"
    elif "_vs_" in gid_lower:  # For cross-variograms
        type_part, vars_part = group_id_label.split(
            "_", 1
        )  # "cross_variogram", "Au_vs_Cu"
        var_names = vars_part.split("_vs_")
        tail_var_disp = var_names[0].replace("_", " ").title()
        head_var_disp = var_names[1].replace("_", " ").title()

        type_disp = type_part.replace("_", " ").title()  # "Cross Variogram"

        y_axis_label = f"{type_disp} $\gamma(h)$ for {tail_var_disp} vs {head_var_disp}"
        title_group_part = f"{type_disp} for {tail_var_disp} and {head_var_disp}"
    # Add more elif for other specific types if needed (e.g., "indicator_var_")
    # Fallback is the generic title_group_part initialized earlier.

    ax.set_xlabel("Lag Distance (h)", fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=12)
    ax.set_title(
        f"Experimental Variograms for {title_group_part} of {os.path.splitext(out_filename_base)[0]}",
        fontsize=13,
    )

    ax.legend(
        title="Directions & Pair Counts", loc="best", fontsize="medium", frameon=True
    )
    ax.grid(True, linestyle=":", alpha=0.5)

    if max_gamma_overall > 1e-9:
        ax.set_ylim(
            bottom=min(0, -0.05 * max_gamma_overall), top=max_gamma_overall * 1.15
        )
    else:
        ax.set_ylim(bottom=-0.01, top=0.15 if max_gamma_overall <= 1e-9 else 1.1)
    ax.set_xlim(left=0)

    plot_filename = os.path.join(
        save_dir, f"{out_filename_base}_{group_id_label}_plot.png"
    )
    try:
        plt.tight_layout()
        fig.savefig(plot_filename, dpi=150)
        print(f"  Plot for {group_id_label} saved to: {plot_filename}")
    except Exception as e:
        print(f"  Error saving plot {plot_filename}: {e}")

    if display_plot:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


if __name__ == "__main__":
    OUT_FILES_DIRECTORY = "."
    PLOTS_SAVE_SUBDIRECTORY = "variogram_plots"  # Changed sub-directory
    DISPLAY_PLOTS_ON_SCREEN = False

    plots_save_path = os.path.join(OUT_FILES_DIRECTORY, PLOTS_SAVE_SUBDIRECTORY)
    try:
        os.makedirs(plots_save_path, exist_ok=True)
        print(f"Plots will be saved in: {os.path.abspath(plots_save_path)}")
    except OSError as oe:
        print(
            f"Error creating directory '{plots_save_path}': {oe}. Plots in current dir."
        )
        plots_save_path = OUT_FILES_DIRECTORY

    try:
        out_file_list = sorted(
            [
                f
                for f in os.listdir(OUT_FILES_DIRECTORY)
                if f.lower().endswith(".out")
                # Optional: keep or remove the specific filename filter if your .out files vary
                # and ("_nlag" in f.lower() and "_xlag" in f.lower())
            ]
        )
    except FileNotFoundError:
        print(f"Error: Dir '{os.path.abspath(OUT_FILES_DIRECTORY)}' not found.")
        out_file_list = []

    if not out_file_list:
        print(
            # f"No .out files with '_nlag' and '_xlag' found in '{os.path.abspath(OUT_FILES_DIRECTORY)}'."
            f"No .out files found in '{os.path.abspath(OUT_FILES_DIRECTORY)}'."
        )
    else:
        print(f"Found {len(out_file_list)} .out files to visualize.")

    for out_filename in out_file_list:
        print(f"\n>>> Processing and plotting: {out_filename} <<<")
        full_file_path = os.path.join(OUT_FILES_DIRECTORY, out_filename)

        parsed_all_groups_data = parse_gamv_output_universal(full_file_path)

        if parsed_all_groups_data:
            base_name_for_plot = os.path.splitext(out_filename)[0]
            print(
                f"  DEBUG: Plotting for file {out_filename}. Parsed groups: {list(parsed_all_groups_data.keys())}"
            )
            for (
                group_key_name,
                directions_data_for_group,
            ) in parsed_all_groups_data.items():
                print(f"  Plotting for Group: {group_key_name}")
                plot_experimental_variograms_universal(
                    directions_data_for_group,
                    group_key_name,
                    base_name_for_plot,
                    save_dir=plots_save_path,
                    display_plot=DISPLAY_PLOTS_ON_SCREEN,
                )
        else:
            print(f"  Could not parse any valid data from '{out_filename}'.")

    print("\nAll .out file visualization attempts completed.")
    if out_file_list and os.path.exists(plots_save_path):
        print(f"Plots are saved in: {os.path.abspath(plots_save_path)}")
