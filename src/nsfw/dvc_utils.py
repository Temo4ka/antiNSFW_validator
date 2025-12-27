import pathlib

from dvc.repo import Repo


def ensure_data_downloaded(data_dir="src/dataset", use_dvc=True):
    data_path = pathlib.Path(data_dir)

    nsfw_path = data_path / "nsfw"
    sfw_path = data_path / "sfw"

    if nsfw_path.exists() and sfw_path.exists():
        nsfw_files = list(nsfw_path.glob("*.jpg"))
        sfw_files = list(sfw_path.glob("*.jpg"))
        if nsfw_files and sfw_files:
            print("Данные уже присутствуют")
            return True

    if use_dvc and Repo is not None:
        print("Загрузка данных через DVC...")
        repo = Repo(".")
        repo.pull()
        print("Данные успешно загружены через DVC")
        return True
    else:
        print("DVC недоступен...")

    return True
