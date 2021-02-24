"""unzip the dataset"""
import zipfile

def main():
    with zipfile.ZipFile("img_align_celeba.zip","r") as zip_ref:
        zip_ref.extractall()
        
if __name__ == "__main__":
    main()
    