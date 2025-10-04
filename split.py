import os
import shutil

INPUT1 = "/kaggle/working/Manga/Input1"
INPUT2 = "/kaggle/working/Manga/Input2"

def is_chapter_folder(path):
    # A "chapter folder" is one that contains at least one image file
    exts = (".jpg", ".jpeg", ".png", ".webp")
    return any(f.lower().endswith(exts) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))

def collect_chapters(root):
    chapter_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        if is_chapter_folder(dirpath):
            chapter_paths.append(dirpath)
    return sorted(chapter_paths)

def split_chapters(chapters):
    mid = len(chapters) // 2
    return chapters[:mid], chapters[mid:]

def move_chapters(chapters, src_root, dst_root):
    for chapter in chapters:
        rel_path = os.path.relpath(chapter, src_root)
        dst_path = os.path.join(dst_root, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.move(chapter, dst_path)
        print(f"Moved {chapter} -> {dst_path}")

if __name__ == "__main__":
    os.makedirs(INPUT2, exist_ok=True)

    chapters = collect_chapters(INPUT1)
    print(f"Found {len(chapters)} chapters")

    half1, half2 = split_chapters(chapters)

    print(f"Assigning {len(half1)} chapters to Input1, {len(half2)} to Input2")

    move_chapters(half2, INPUT1, INPUT2)

    print("Split complete.")
