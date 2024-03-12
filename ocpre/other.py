import os

def del_file(file_path):
    # 指定要删除的文件路径
    # 尝试删除文件
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"The file {file_path} was successfully deleted ")
        except OSError as e:
            print(f"Error deleting file {file_path}: {e}")
    else:
        print(f" {file_path} does not exist")
        

def exist_folder(judge_folder_path, is_creat=False):
    """_summary_
        Check if the folder exists. If the folder is wanted to be created,
        'is_creat' should be seted to True.
    Args:
        judge_folder_path (_type_): _description_
        is_creat (bool, optional): _description_. Defaults to False.

    Raises:
        FileNotFoundError: _description_
    """    
    if os.path.exists(judge_folder_path):
        print(f"folder '{judge_folder_path}' exists")
    elif is_creat:
        os.makedirs(judge_folder_path)
        print(f"folder '{judge_folder_path}' is created")
    else:
        raise FileNotFoundError(f"folder '{judge_folder_path}' does not exist, make sure the calculation has ended")

