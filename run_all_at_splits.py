import os
import pandas as pd
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Nota: As sementes aleatórias são configuradas em run_full_pipeline.py -> main()
# Cada invocação do pipeline via os.system() vai chamar set_random_seeds() automaticamente

# read CSV files in dataset_splits directory and return a list of dictionaries with the split information (train, val, test directories)
def load_dataset_splits(splits_dir: str = "dataset_splits/") -> list:
    """
    Carrega os arquivos CSV de splits e retorna uma lista de dicionários com as informações dos splits.
    
    Args:
        splits_dir (str): Diretório onde estão os arquivos CSV de splits
    
    Returns:
        list: Lista de dicionários com as informações dos splits
    """
    splits = []
    for filename in os.listdir(splits_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(splits_dir, filename)
            split_df = pd.read_csv(filepath)
            splits.extend(split_df.to_dict(orient="records"))

    differentiation_degree_dict = {
        "Moderately differentiated": "tubular-mod_dif",
        "Well-differentiated": "tubular-bem_dif",
        "Poorly differentiated": "tubular-mal_dif"
    }

    # change differentiation values in splits to the corresponding values in the dictionary
    for split in splits:
        if split["differentiation"] in differentiation_degree_dict:
            split["differentiation"] = differentiation_degree_dict[split["differentiation"]]

    return splits

def tubular_dataset_split_create(n_split: int = 0, data_dict: dict = None) -> None:
    """
    Cria os splits do dataset tubular a partir de um dicionário com as informações dos splits.
    
    Args:
        n_split (int): Número do split a ser criado
        data_dict (dict): Dicionário com as informações dos splits
    Returns:
        None
    """
    if data_dict is None:
        raise ValueError("O dicionário de dados não pode ser None.")
    
    
    split_name = f"split_0{n_split}"
    output_dir = f"tubular-subtype-molecular-dataset_{split_name}"

    # create train,val and test directories for the split
    for subset in ["train", "val", "test"]:
        os.makedirs(f"{output_dir}/{subset}/", exist_ok=True)
    
    # move the files to the corresponding directories according to the split information in the data_dict
    for item in data_dict:
        src_path = f'patches/{item["category"]}/{item["slide_id"]}/{item["differentiation"]}'
        dst_path = f"{output_dir}/{item['original_split']}/{item['category']}"
        for filename in os.listdir(src_path):
            # copy  the file to the corresponding directory
            os.makedirs(dst_path, exist_ok=True)
            os.system(f"cp {src_path}/{filename} {dst_path}/{filename}")
            # print(f"cp {src_path}/{filename} {dst_path}/{filename}")
            pass

if __name__ == "__main__":
    
    for n_split in range(5,10):
        logger.info("="*80)
        logger.info(f"INICIANDO PIPELINE DE TREINAMENTOS - split_0{n_split}")
        logger.info("="*80)
        path = f"dataset_splits/split_0{n_split}"
        splits = load_dataset_splits(path)
        # print(f"Found {len(splits)} splits:")
        tubular_dataset_split_create(n_split, splits)

        # runs all networks for split datasets created with parameters:
        # --all
        # --epochs 100
        # --base_learning_rate 0.0001
        # --output_dir outputs/split_0{n_split}
        # --base_path tubular-subtype-molecular-dataset_split_0{n_split}/
        os.system(f"python run_full_pipeline.py --all --epochs 100 --base_lr 0.0001 --output_dir outputs/split_0{n_split} --base_path tubular-subtype-molecular-dataset_split_0{n_split}/")
        
        os.system(f"python run_full_pipeline.py --all --epochs 100 --base_lr 0.001 --output_dir outputs/split_0{n_split} --base_path tubular-subtype-molecular-dataset_split_0{n_split}/")

        # delete tubular-subtype-molecular-dataset_{split_name} directories after running the pipeline for all networks
        os.system(f"rm -rf tubular-subtype-molecular-dataset_split_0{n_split}")