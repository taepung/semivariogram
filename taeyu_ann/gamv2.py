import os
import subprocess
import datetime


def run_all_gamv_in_directory_no_log(par_directory, gamv_exe_path):
    """
    지정된 디렉토리의 모든 .par 파일을 찾아 GSLIB gamv.exe로 실행합니다.
    gamv.exe는 파라미터 파일명을 표준 입력으로 받습니다.
    gamv.exe의 출력은 실시간으로 콘솔에 표시되며, 로그 파일은 생성하지 않습니다.

    Args:
        par_directory (str): .par 파일들이 들어있는 디렉토리 경로.
        gamv_exe_path (str): gamv.exe 실행 파일의 전체 경로 또는 PATH에 등록된 경우 "gamv" / "gamv.exe".
    """
    print(
        f"Starting GAMV.EXE execution for .par files in: {os.path.abspath(par_directory)}"
    )
    print(f"GAMV Executable to be used (from CWD or PATH): {gamv_exe_path}")
    print("-" * 70)

    try:
        par_files = sorted(
            [f for f in os.listdir(par_directory) if f.lower().endswith(".par")]
        )
    except FileNotFoundError:
        error_msg = f"Error: The directory '{par_directory}' was not found."
        print(error_msg)
        return

    total_files = len(par_files)
    success_count = 0
    failure_count = 0

    if total_files == 0:
        no_files_msg = "No .par files found in the specified directory."
        print(no_files_msg)
        return

    print(
        f"Found {total_files} .par files to process in '{os.path.abspath(par_directory)}'.\n"
    )

    for i, par_filename in enumerate(par_files):
        start_time_dt = datetime.datetime.now()
        start_time_str = start_time_dt.strftime("%Y-%m-%d %H:%M:%S")

        log_entry_prefix = f"[{i+1}/{total_files}] Processing: {par_filename}"
        print(log_entry_prefix)
        print(f"  -> Started at: {start_time_str}")
        print(
            f"    Attempting to run: {gamv_exe_path} (will provide '{par_filename}' via stdin)"
        )
        print("-" * 20 + " GAMV Output Starts " + "-" * 20)

        try:
            # gamv.exe를 실행하고, input 인자를 통해 파라미터 파일명을 표준 입력으로 전달합니다.
            # capture_output은 False (기본값)이므로 stdout/stderr가 터미널로 직접 향합니다.
            process = subprocess.run(
                [gamv_exe_path],  # 명령어만 전달
                input=par_filename
                + "\n",  # 파라미터 파일명 + 엔터키를 표준 입력으로 전달
                text=True,  # input을 문자열로 처리, stdout/stderr도 텍스트로 (만약 캡처한다면)
                check=False,  # returncode가 0이 아니어도 예외 발생 안 함
                cwd=par_directory,  # 작업 디렉토리 설정
                encoding="utf-8",  # GSLIB 출력 인코딩에 따라 (예: 'cp949' for Korean Windows)
                errors="replace",  # 인코딩 오류 발생 시 대체 문자로 처리
                # stdout=None, # 명시하지 않으면 부모의 stdout (터미널)으로 연결됨 (capture_output=False 시)
                # stderr=None  # 명시하지 않으면 부모의 stderr (터미널)으로 연결됨 (capture_output=False 시)
            )

            # gamv.exe의 출력이 끝난 후 이 부분이 실행됩니다.
            print("-" * 20 + " GAMV Output Ends " + "-" * 22)
            end_time_dt = datetime.datetime.now()
            duration_seconds = (end_time_dt - start_time_dt).total_seconds()

            print(
                f"  -> subprocess.run for '{par_filename}' finished at: {end_time_dt.strftime('%Y-%m-%d %H:%M:%S')} (Duration: {duration_seconds:.2f}s)"
            )

            # capture_output이 False(또는 생략)이므로 process.stdout과 process.stderr는 None입니다.
            # gamv.exe의 출력은 이미 실시간으로 터미널에 표시되었습니다.

            if process.returncode == 0:
                print(f"  -> Status: Success (Return Code: {process.returncode})")
                success_count += 1
            else:
                print(f"  -> Status: FAILED (Return Code: {process.returncode})")
                # 실패 시 gamv.exe가 출력한 오류 메시지는 이미 터미널에 표시되었을 것입니다.
                failure_count += 1

        except FileNotFoundError:
            error_msg = (
                f"CRITICAL ERROR: The GAMV executable '{gamv_exe_path}' was not found."
            )
            print(f"  -> {error_msg}")
            print("Aborting further processing.")
            failure_count += total_files - i
            break
        except Exception as e:
            error_msg = (
                f"PYTHON SCRIPT ERROR occurred while processing {par_filename}: {e}"
            )
            print(f"  -> {error_msg}")
            failure_count += 1
        print("-" * 70)  # 각 파일 처리 후 명확한 구분선

    summary_msg = (
        f"\n--- GAMV Execution Session Summary ---\n"
        f"Total .par files found: {total_files}\n"
        f"Attempted to process: {i + 1 if total_files > 0 and i < total_files else total_files if total_files > 0 else 0}\n"
        f"Successfully processed: {success_count}\n"
        f"Failed to process: {failure_count}\n"
        f"--- Session Ended at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
    )
    print(summary_msg)


if __name__ == "__main__":
    # ==============================================================================
    # 사용자 설정 부분
    # 현재 스크립트가 실행되는 폴더에 gamv.exe와 .par 파일들이 있다고 가정합니다.
    # ==============================================================================
    # gamv.exe가 현재 폴더에 있으므로 파일명만 지정 (Windows의 경우 확장자 포함)
    # 또는 시스템 PATH에 등록되어 있다면 이것만으로도 충분합니다.
    GAMV_EXECUTABLE = "gamv.exe"

    # .par 파일들도 현재 폴더에 있으므로 "." (현재 디렉토리)로 지정
    PARAMS_DIRECTORY = "."
    # ==============================================================================

    if not os.path.isdir(PARAMS_DIRECTORY):
        print(
            f"Error: The specified .par directory '{os.path.abspath(PARAMS_DIRECTORY)}' does not exist or is not a directory."
        )
        print("Please check the PARAMS_DIRECTORY variable in the script.")
    elif not GAMV_EXECUTABLE:
        print("Error: GAMV_EXECUTABLE path is not specified.")
    else:
        run_all_gamv_in_directory_no_log(
            os.path.abspath(PARAMS_DIRECTORY), GAMV_EXECUTABLE
        )
