import os
import subprocess



def convert_all_webm_in_directory(directory):
    """
    Converts all .webm files in the specified directory to .mp4.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.webm'):
            input_file, extension = filename.split(".")
            print(filename, input_file, extension , "----------")
            os.rename(f'/home/siddhi/Downloads/20bn-something-something-v2/{filename}',f'/home/siddhi/Downloads/20bn-something-something-v2/{input_file}.mp4')

if __name__ == "__main__":
    # Replace with the directory where your .webm files are located
    directory = '/home/siddhi/Downloads/20bn-something-something-v2'  # Change to your directory
    convert_all_webm_in_directory(directory)

