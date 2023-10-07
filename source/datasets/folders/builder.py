import pandas as pd


class FolderBuilder:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe

    def build_folder(self, images_folder, images_folder_target):
        if not os.path.exists(images_folder_target):
            os.mkdir(images_folder_target, 0o755)

        for label in tqdm(self.df["label"].value_counts().index):
            if not os.path.exists(f"{images_folder_target}/{label}"):
                os.mkdir(f"{images_folder_target}/{label}", 0o755)

            for image in self.df[self.df["label"] == label].image_lien:
                source = f"{images_folder}/{image}"
                dest = f"{images_folder_target}/{label}/{image}"

                try:
                    if os.path.exists(source) and not os.path.exists(dest):
                        shutil.copyfile(source, dest)
                except FileNotFoundError as e:
                    pass
                    print(e)
