import pandas as pd
from matplotlib import pyplot as plt
import argparse
from copy import deepcopy


def get_speedup_ratio(is_static=True, is_splitk=False, type: str="all"):
    df = pd.read_csv(f"testing/python/dynamic/.log/merged_results_{'static' if is_static else 'dynamic'}{'_splitk' if is_splitk else ''}.csv")

    # Choose the rows when type != "all"
    if type == "power2":
        df = df[df["type"] == "power2"]
    elif type == "random":
        df = df[df["type"] == "random"]
    elif type == "llm":
        df = df[df["type"] != "power2"]
        df = df[df["type"] != "random"]
    
    good_speedup_df = df[df["speedup"] >= 1]
    medium_speedup_df = df[df["speedup"] >= 0.95]
    borderline_speedup_df = df[df["speedup"] >= 0.9]

    # Calculate the ratio of the number of rows in the dataframe to the number of rows in the good_speedup_df, medium_speedup_df, borderline_speedup_df
    good_speedup_ratio = len(good_speedup_df) / len(df)
    medium_speedup_ratio = len(medium_speedup_df) / len(df)
    borderline_speedup_ratio = len(borderline_speedup_df) / len(df)
    print(f"type: {type}, is_static: {is_static}, is_splitk: {is_splitk}")
    print(f"good_speedup_ratio: {good_speedup_ratio}\nmedium_speedup_ratio: {medium_speedup_ratio}\nborderline_speedup_ratio: {borderline_speedup_ratio}")

    return good_speedup_ratio, medium_speedup_ratio, borderline_speedup_ratio


def analysis_speedup(is_static=True, is_splitk=False, type: str="all"):
    df = pd.read_csv(f"testing/python/dynamic/.log/merged_results_{'static' if is_static else 'dynamic'}{'_splitk' if is_splitk else ''}.csv")

    # Choose the rows when type != "all"
    if type == "power2":
        df = df[df["type"] == "power2"]
    elif type == "random":
        df = df[df["type"] == "random"]
    elif type == "llm":
        df = df[df["type"] != "power2"]
        df = df[df["type"] != "random"]
    
    # Filter the dataframe to only include the rows where the speedup is greater than 0.9
    if type == "all" or type == "power2":
        df = df[df["speedup"] >= 0.9]

    # Get the speedup from the dataframe and create a bar plot without the first 1 row
    if type == "all" or type == "power2":
        speedup = df["speedup"][1:]
        x_labels = [f"({row['M']}, {row['N']}, {row['K']}, {row['type']})" for _, row in df.iterrows()][1:]
    else:
        speedup = df["speedup"]
        x_labels = [f"({row['M']}, {row['N']}, {row['K']}, {row['type']})" for _, row in df.iterrows()]
    
    # Define color ranges and corresponding colors
    colors = []
    for s in speedup:
        if s >= 1:
            colors.append('green')  # Good performance
        elif s >= 0.95:
            colors.append('yellow')  # Slightly worse
        elif s >= 0.9:
            colors.append('orange')  # Worse
        else:
            colors.append('red')  # Bad performance

    plt.figure(figsize=(20, 10))  # Set figure size to be larger
    bars = plt.bar(range(len(speedup)), speedup, color=colors)
    plt.xlabel("(M, N, K, type)")
    plt.xticks(range(len(speedup)), x_labels, rotation=45, ha='right')
    plt.ylabel("Speedup")
    plt.title(f"Speedup of {type} over cublas")
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()  # Adjust layout to prevent label cutoff
    plt.savefig(f"testing/python/dynamic/.log/speedup_{'static' if is_static else 'dynamic'}{'_splitk' if is_splitk else ''}_{type}.png")
    plt.close()


def analysis_speedup_bad_config(is_static=True, is_splitk=False, type: str="all"):
    df = pd.read_csv(f"testing/python/dynamic/.log/merged_results_{'static' if is_static else 'dynamic'}{'_splitk' if is_splitk else ''}.csv")

    # Choose the rows when type != "all"
    if type == "power2":
        df = df[df["type"] == "power2"]
    elif type == "random":
        df = df[df["type"] == "random"]
    elif type == "llm":
        df = df[df["type"] != "power2"]
        df = df[df["type"] != "random"]
    
    # Filter the dataframe to only include the rows where the speedup is less than 0.9
    original_df = deepcopy(df)
    df = df[df["speedup"] < 0.9]

    # Get the speedup from the dataframe
    speedup = df["speedup"]
    x_labels = [f"({row['M']}, {row['N']}, {row['K']}, {row['type']})" for _, row in df.iterrows()]
    
    # Analysis the bad config
    if type == "power2":
        # Calculate the ratio of the average of M, N, K is less than power of 2
        for i in range(9, 15):
            df_average_less = df[df["M"] * df["N"] * df["K"] <= 2**(3*i)]
            print(f"average of M, N, K is less than 2^{i} ratio in only bad config: {len(df_average_less) / len(df)}")
        # Calculate the ratio of the min of M, N, K is less than power of 2
        for i in range(9, 15):
            df_min_less = df[(df["M"] <= 2**(i)) | (df["N"] <= 2**(i)) | (df["K"] <= 2**(i))]
            print(f"min of M, N, K is less than 2^{i} ratio in only bad config: {len(df_min_less) / len(df)}")
    elif type == "llm":
        # Calculate the ratio of prefill and decode
        df_prefill = df[df["M"] != 1]
        df_decode = df[df["M"] == 1]
        print(f"prefill ratio: {len(df_prefill) / len(df)}")
        print(f"decode ratio: {len(df_decode) / len(df)}")
        # Calculate the ratio of model type
        df_llama = df[df["type"] == "llama_linear"]
        df_gpt = df[df["type"] == "gpt2_conv1d"]
        df_qwen = df[df["type"] == "qwen_linear"]
        print(f"llama ratio in only bad config: {len(df_llama) / len(df)}")
        print(f"gpt ratio in only bad config: {len(df_gpt) / len(df)}")
        print(f"qwen ratio in only bad config: {len(df_qwen) / len(df)}")

    
def analysis_speedup_good_config(is_static=True, is_splitk=False, type: str="all"):
    df = pd.read_csv(f"testing/python/dynamic/.log/merged_results_{'static' if is_static else 'dynamic'}{'_splitk' if is_splitk else ''}.csv")

    # Choose the rows when type != "all"
    if type == "power2":
        df = df[df["type"] == "power2"] 
    elif type == "random":
        df = df[df["type"] == "random"]
    elif type == "llm":
        df = df[df["type"] != "power2"]
        df = df[df["type"] != "random"]
    
    # Filter the dataframe to only include the rows where the speedup is greater than 0.9
    df = df[df["speedup"] >= 0.9]

    # Group by block sizes and count their frequency
    block_sizes_df = df.groupby(['block_M', 'block_N', 'block_K']).size().reset_index(name='frequency')
    # Sort by frequency in descending order
    block_sizes_df = block_sizes_df.sort_values('frequency', ascending=False)    
    print("\nBlock size configurations and their frequencies:")
    for _, row in block_sizes_df.iterrows():
        print(f"block_M={row['block_M']}, block_N={row['block_N']}, block_K={row['block_K']}: {row['frequency']} occurrences")

    # Group by block sizes and threads and count their frequency
    block_thread_df = df.groupby(['block_M', 'block_N', 'block_K', 'thread_num']).size().reset_index(name='frequency')
    block_thread_df = block_thread_df.sort_values('frequency', ascending=False)
    print("\nBlock size and thread configurations and their frequencies:")
    for _, row in block_thread_df.iterrows():
        print(f"block_M={row['block_M']}, block_N={row['block_N']}, block_K={row['block_K']}, thread_num={row['thread_num']}: {row['frequency']} occurrences")

    # Group by all configurations and count their frequency
    all_config_df = df.groupby(['block_M', 'block_N', 'block_K', 'thread_num', 'num_stages', 'enable_rasteration']).size().reset_index(name='frequency')
    all_config_df = all_config_df.sort_values('frequency', ascending=False)
    print("\nAll configurations and their frequencies:")
    for _, row in all_config_df.iterrows():
        print(f"block_M={row['block_M']}, block_N={row['block_N']}, block_K={row['block_K']}, thread_num={row['thread_num']}, num_stages={row['num_stages']}, enable_rasteration={row['enable_rasteration']}: {row['frequency']} occurrences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_static", type=bool, default=True)
    parser.add_argument("--is_splitk", type=bool, default=False)
    parser.add_argument("--type", type=str, default="all")
    args = parser.parse_args()
    get_speedup_ratio(is_static=args.is_static, is_splitk=args.is_splitk, type=args.type)
    analysis_speedup(is_static=args.is_static, is_splitk=args.is_splitk, type=args.type)
    analysis_speedup_bad_config(is_static=args.is_static, is_splitk=args.is_splitk, type=args.type)
    analysis_speedup_good_config(is_static=args.is_static, is_splitk=args.is_splitk, type=args.type)
