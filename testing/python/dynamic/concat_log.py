import os
import pandas as pd
import glob


def concat_csv_files(type: str="static", is_splitk=False):
    # Get all CSV files starting with "best_config_static"
    csv_files = glob.glob(f"testing/python/dynamic/.log/best_config_{type}_*.csv")
    
    # Read and concatenate all CSV files
    dfs = []
    for file in csv_files:
        if (not is_splitk) and "splitk" in file:
            continue
        elif is_splitk and "splitk" not in file:
            continue
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Concatenate all dataframes and sort by M, N, K
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(['M', 'N', 'K'])
    
    # Save the combined dataframe to a new CSV file
    if is_splitk:
        combined_df.to_csv(f"testing/python/dynamic/.log/best_configs_{type}_splitk_all.csv", index=False)
        print(f"Combined {len(csv_files)} CSV files into best_configs_{type}_splitk_all.csv")
    else:
        combined_df.to_csv(f"testing/python/dynamic/.log/best_configs_{type}_all.csv", index=False)
        print(f"Combined {len(csv_files)} CSV files into best_configs_{type}_all.csv")


def df_with_cublas_all(type: str="static", is_splitk=False):
    tl_df = pd.read_csv(f"testing/python/dynamic/.log/best_configs_{type}_all.csv")
    cublas_df = pd.read_csv(f"testing/python/dynamic/.log/cublas_results.csv")

    # Merge the two dataframes on the 'M', 'N', 'K' columns
    merged_df = pd.merge(cublas_df, tl_df, on=['M', 'N', 'K'], how='left')

    # Merge the merged_df with the matmul_config_df on the 'M', 'N', 'K' columns
    matmul_config_df = pd.read_csv(f"testing/python/dynamic/.log/matmul_config_all.csv")
    merged_df = pd.merge(merged_df, matmul_config_df, on=['M', 'N', 'K'], how='left')

    # Rename the columns
    merged_df = merged_df.rename(columns={
        "time(ms)": "cublas_time",
        "best_latency": f"{type}_time",
        "TFLOPS_x": "cublas_TFLOPS",
        "TFLOPS_y": f"{type}_TFLOPS",
    })

    # Switch the order of the columns
    if is_splitk:
        merged_df = merged_df[['M', 'N', 'K', 'block_M', 'block_N', 'block_K', 'num_stages', 'thread_num', 'enable_rasteration', 'split_k', 'cublas_time', f"{type}_time", 'cublas_TFLOPS', f"{type}_TFLOPS", "type"]]
    else:
        merged_df = merged_df[['M', 'N', 'K', 'block_M', 'block_N', 'block_K', 'num_stages', 'thread_num', 'enable_rasteration', 'cublas_time', f"{type}_time", 'cublas_TFLOPS', f"{type}_TFLOPS", "type"]]

    # Compute the speedup of TL over cublas and sort by speedup
    merged_df["speedup"] = merged_df["cublas_time"] / merged_df[f"{type}_time"]
    merged_df = merged_df.sort_values(by="speedup", ascending=False)

    # Output the result to a csv file
    merged_df.to_csv(f"testing/python/dynamic/.log/merged_results_{type}.csv", index=False)


def df_with_llm_config(type: str="static", is_splitk=False):
    merged_df = pd.read_csv(f"testing/python/dynamic/.log/merged_results_{type}{'_splitk' if is_splitk else ''}.csv")

    # Choose the rows with type != "power2" and type != "random"
    llm_df = merged_df[merged_df["type"] != "power2"]
    llm_df = llm_df[llm_df["type"] != "random"]
    llm_df.to_csv(f"testing/python/dynamic/.log/llm_results_{type}{'_splitk' if is_splitk else ''}.csv", index=False)


def df_with_power2(type: str="static", is_splitk=False):
    merged_df = pd.read_csv(f"testing/python/dynamic/.log/merged_results_{type}{'_splitk' if is_splitk else ''}.csv")

    # Choose the rows with type == "power2"
    power2_df = merged_df[merged_df["type"] == "power2"]
    power2_df.to_csv(f"testing/python/dynamic/.log/power2_results_{type}{'_splitk' if is_splitk else ''}.csv", index=False)


if __name__ == "__main__":
    concat_csv_files(type="static", is_splitk=False)
    df_with_cublas_all(type="static", is_splitk=False)
    df_with_llm_config(type="static", is_splitk=False)
    df_with_power2(type="static", is_splitk=False)
