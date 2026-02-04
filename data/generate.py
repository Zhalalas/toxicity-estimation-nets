import pandas as pd
import numpy as np
import argparse

# Use argparse to accept parameters such as sugar_n min and max values, with defaults, and min-max hours
# Use argparse to accept the number of entries to generate, with a default of 2000
# Use argparse to accept the output file path, with a default of 'datasets/synthetic_data.csv'

parser = argparse.ArgumentParser(description='Generate synthetic data for sugar levels.')
parser.add_argument('--sugar_1_min', type=float, default=0, help='Minimum value for sugar_1')
parser.add_argument('--sugar_1_max', type=float, default=5, help='Maximum value for sugar_1')
parser.add_argument('--sugar_2_min', type=float, default=0, help='Minimum value for sugar_2')
parser.add_argument('--sugar_2_max', type=float, default=5, help='Maximum value for sugar_2')
parser.add_argument('--sugar_3_min', type=float, default=0, help='Minimum value for sugar_3')
parser.add_argument('--sugar_3_max', type=float, default=5, help='Maximum value for sugar_3')
parser.add_argument('--sugar_4_min', type=float, default=0, help='Minimum value for sugar_4')
parser.add_argument('--sugar_4_max', type=float, default=5, help='Maximum value for sugar_4')
parser.add_argument('--hours_min', type=int, default=0, help='Minimum hours value')
parser.add_argument('--hours_max', type=int, default=336, help='Maximum hours value')
parser.add_argument('--num_entries', type=int, default=2000, help='Number of entries to generate')
parser.add_argument('--output_file', type=str, default='datasets/synthetic_data.csv', help='Output file path')

args = parser.parse_args()

def main(args: argparse.Namespace):

    sugar_1=np.random.uniform(args.sugar_1_min, args.sugar_1_max, args.num_entries)
    sugar_2=np.random.uniform(args.sugar_2_min, args.sugar_2_max, args.num_entries)
    sugar_3=np.random.uniform(args.sugar_3_min, args.sugar_3_max, args.num_entries)
    sugar_4=np.random.uniform(args.sugar_4_min, args.sugar_4_max, args.num_entries)
    hours=np.random.randint(args.hours_min, args.hours_max + 1, args.num_entries)

    data = pd.DataFrame({
        'sugar_1': sugar_1,
        'sugar_2': sugar_2,
        'sugar_3': sugar_3,
        'sugar_4': sugar_4,
        'hours': hours
    })
    toxic_deg=(sugar_1*1.1 + sugar_2 + sugar_3*0.9 + sugar_4*0.8) / (5*1.1 + 5 + 5*0.9 + 5*0.8) +  0.85 * (hours / (14*24))
    data['toxic_deg'] = toxic_deg
    labels = (toxic_deg > 0.65).astype(int)
    data['label'] = labels
    data.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main(args)