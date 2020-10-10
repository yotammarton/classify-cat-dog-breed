"""
code for creating dataframes .csv with image paths for models training and testing
"""

import pandas as pd
import os


def oxford_cats_dogs_images():
    """
    path      | cat/dog  | breed  | dataset
    ___________________________________
    file path | dog     | 3      | oxford
    file path | cat     | 6      | oxford
    :return: data frame with oxford images paths
    """
    df = pd.DataFrame(columns=['path', 'cat/dog', 'breed', 'dataset'])
    path = os.path.join('annotations', 'list.txt')

    with open(path) as file:
        for line in file.readlines():
            if line[0] == '#':
                continue
            splited = line.split()
            file_name, breed, cat_or_dog = splited[0] + '.jpg', splited[1], splited[2]
            file_path = os.path.join('images', file_name)
            df = df.append({'path': file_path, 'cat/dog': 'cat' if cat_or_dog == '1' else 'dog',
                            'breed': breed, 'dataset': 'oxford'}, ignore_index=True)

    return df[['path', 'cat/dog', 'breed', 'dataset']]


def train_test_split(df):
    # get 70% of breed images for train, 10% validation and the rest 200% for test
    # roughly 200 images per breed
    result_df = None
    for breed in set(df['breed']):
        # get the df for specific breed and shuffle it
        breed_df = df[df['breed'] == breed].sample(frac=1, random_state=42)
        test_size = int(len(breed_df) * 0.2)
        validation_size = int(len(breed_df) * 0.1)
        train_size = len(breed_df) - test_size - validation_size
        breed_df['train/test'] = ['test'] * test_size + ['train'] * train_size + ['validation'] * validation_size
        if result_df is None:
            result_df = breed_df.copy()
        else:
            result_df = result_df.append(breed_df)
    return result_df.reset_index(drop=True)


if __name__ == '__main__':
    """build the .csv for image paths for breed classification model"""
    df_oxford = oxford_cats_dogs_images()
    df_oxford = train_test_split(df_oxford)
    df_oxford.to_csv(f"data_paths_and_classes_{'windows' if os.name == 'nt' else 'unix'}.csv")

    pass
