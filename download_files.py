import os
import subprocess


def download_files_from_folder(folder_url, download_dir):
    """
    Downloads files starting with 18 to 43 and ending with 'mask.png' using wget command.
    """
    # Ensure the directory exists
    fragment_id = folder_url.split("/")[-1]
    folder_url_layers = f"{folder_url}/layers"
    download_dir_layers = f"{download_dir}/layers"

    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(download_dir_layers, exist_ok=True)

    # List of files to download
    file_list = []

    # Prepare the wget command for files starting with 18 to 43 and 'mask.png'
    wget_command = f"wget -r -np -nc -nd -A'18.*,19.*,20.*,21.*,22.*,23.*,24.*,25.*,26.*,27.*,28.*,29.*,30.*,31.*,32.*,33.*,34.*,35.*,36.*,37.*,38.*,39.*,40.*,41.*,42.*,43.*' {folder_url_layers} -P {download_dir_layers}"
    wget_command_1 = f"wget {folder_url}/{fragment_id}_flat_mask.png -P {download_dir}"

    try:
        # Execute wget command
        subprocess.run(wget_command, shell=True, check=True)
        subprocess.run(wget_command_1, shell=True, check=True)
        print(f"Downloaded files from {folder_url} to {download_dir}")

        # After downloading, return the list of downloaded files
        for root, _, files in os.walk(download_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        return file_list
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while downloading from {folder_url}: {e}")
        return []


def run_preprocessing_script(folder_path):
    """
    Runs the preprocessing script on the downloaded folder.
    """
    # Assuming the preprocessing script is a Python script
    # You can modify this based on how your preprocessing script works
    result = subprocess.run(["python3", "preprocess.py", folder_path], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Preprocessing completed for {folder_path}")
    else:
        print(f"Preprocessing failed for {folder_path}. Error: {result.stderr}")


def process_folders(folders, base_url, download_base_dir):
    """
    Process each folder by downloading files and running preprocessing.
    """
    for folder in folders:
        folder_url = f"{base_url}/{folder}"
        folder_download_dir = os.path.join(download_base_dir, folder)

        # Download the files
        print(f"Downloading files from {folder_url}...")
        downloaded_files = download_files_from_folder(folder_url, folder_download_dir)

        # Run the preprocessing script after downloading
        if downloaded_files:
            print(f"Running preprocessing for {folder}...")
            run_preprocessing_script(folder_download_dir)
        else:
            print(f"No files downloaded for {folder}. Skipping preprocessing.")


if __name__ == "__main__":
    # Example folder names and base URL
    segments = {
        "scroll_5": {
            "base_url": "https://dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/paths",
            "download_base_dir": "/home/sergeipnev/work/VC/data/scroll_5",
            "folders": [
                "20241025150211",
                "20241113070770",
                "20241025145341",
                "20241028121111",
                "20241102125650",
                "20241108111522",
                "20241113080880",
                "20241025145701",
                "20241030152031",
                "20241102160330",
                "20241113090990",
                "20241108115232",
                "20241108120732",
            ]
        }
    }

        # "scroll_1": {
        #     "base_url": "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths",
        #     "download_base_dir": "/home/sergeipnev/work/VC/data/scroll_1",
        #     "folders": [
        #         "20231221180251",
        #         "20231031143852",
        #         "20231022170901",
        #         "20231016151002",
        #         "20231007101619",
        #         "20231012085431",
        #         "20231004222109",
        #         "20231001164029",
        #         "20230909121925",
        #         "20230905134255",
        #         "20230904135535",
        #         "20230827161847",
        #         "20230701020044",
        #         "20230611014200",
        #         "20230522181603",
        #
        #         "20240301161650",
        #         "20240227085920",
        #         "20240223130140",
        #         "20240222111510",
        #         "20240221073650",
        #         "20240218140920",
        #     ],
        # },
    # }

    # segments = {
    #     "scroll_1": {
    #         "base_url": "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/paths",
    #         "download_base_dir": "/home/sergeipnev/work/VC/data/scroll_1",
    #         "folders": [
    #             "20230826170124",
    #             "20231012173610",
    #             "20230929220926",
    #             "20231210121321",
    #             "20230901184804",
    #             "20230530172803",
    #             "20230820203112",
    #             "20230520175435",
    #             "20230702185753",
    #             "20231106155351",
    #             "20230530212931",
    #             "20230522215721",
    #             "20230904020426",
    #             "20230531193658",
    #             "20230620230617",
    #             "20231012184420",
    #             "20230601193301",
    #             "20230902141231",
    #             "20231005123336",
    #             "20230530164535",
    #             "20230903193206",
    #             "20230531121653",
    #             "20230620230619",
    #         ]
    #     },
    #     "scroll_4": {
    #         "base_url": "https://dl.ash2txt.org/full-scrolls/Scroll4/PHerc1667.volpkg/paths",
    #         "download_base_dir": "/home/sergeipnev/work/VC/data/scroll_4",
    #         "folders": [
    #             "20240304161941",
    #             "20231210132040",
    #             "20240304144031",
    #             "20240304141530",
    #             "20240304141531"
    #         ]
    #     },
    #     "scroll_2": {
    #         "base_url": "https://dl.ash2txt.org/full-scrolls/Scroll2/PHercParis3.volpkg/paths",
    #         "download_base_dir": "/home/sergeipnev/work/VC/data/scroll_2",
    #         "folders": [
    #             "20230421192746",
    #             "20230422213203",
    #             "20230503120034",
    #             "20230507064642",
    #             "20230508181757",
    #             "20230512211850",
    #             "20230517000306",
    #             "20230517193901",
    #             "20230519033308",
    #             "20230801194757",
    #             "20240516205750",
    #             "20230421235552",
    #             "20230424181417",
    #             "20230425200944",
    #             "20230427171131",
    #             "20230504151750",
    #             "20230508032834",
    #             "20230509173534",
    #             "20230515151114",
    #             "20230517153958",
    #             "20230517214715",
    #             "20230520080703",
    #             "20230422011040",
    #             "20230424213608",
    #             "20230426114804",
    #             "20230501042136",
    #             "20230506111616",
    #             "20230508171353",
    #             "20230512192835",
    #             "20230516154633",
    #             "20230517171727",
    #             "20230518210035",
    #             "20230522182853",
    #             "20230709155141"
    #         ]
    #     },
    #     "scroll_3": {
    #         "base_url": "https://dl.ash2txt.org/full-scrolls/Scroll3/PHerc332.volpkg/paths",
    #         "download_base_dir": "/home/sergeipnev/work/VC/data/scroll_3",
    #         "folders": [
    #             "20240712074250",
    #             "20240618142020",
    #             "20240712064330",
    #             "20240715203740",
    #             "20240712071520",
    #             "20240716140050"
    #         ]
    #     },
    #
    # }

    for key, item in segments.items():
        base_url = item["base_url"]
        download_base_dir = item["download_base_dir"]
        folders = item["folders"]

        process_folders(folders, base_url, download_base_dir)


    bad_segments_from_scroll_1 = [
        "20231215151901",
        "20231106155350",
        "20231016151000",
        "20231031143850",
        "20231012184423",
        "20231012184421",
        "20231007101615"
    ]