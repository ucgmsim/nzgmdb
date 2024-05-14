from pathlib import Path

from nzgmdb.management import file_structure


def merge_im_data(
    main_dir: Path,
):
    """
    Merge the IM data into a single flatfile
    """
    # Get the flatfile directory
    flatfile_dir = file_structure.get_flatfile_dir(main_dir)
    # Get the IM directory
    im_dir = file_structure.get_im_dir(main_dir)
