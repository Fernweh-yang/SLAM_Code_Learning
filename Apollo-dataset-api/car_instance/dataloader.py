import argparse
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render car instance and convert car labelled files.')
    parser.add_argument('--input_dir',default='../apolloscape/sample/',
                        help='input folder: where store the dataset')
    parser.add_argument('--output_dir',default='../output/',
                        help='output folder: where store the log files')
    
    args = parser.parse_args()
    