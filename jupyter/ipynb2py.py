import os
import nbformat
from nbconvert import PythonExporter


def ipynb2py(jupyter_dir):
    # jupyter 폴더 상위 경로
    base_dir = os.path.abspath(os.path.join(jupyter_dir, ".."))
    src_dir = os.path.join(base_dir, "src")

    # nbconvert PythonExporter 인스턴스
    exporter = PythonExporter()

    for root, dirs, files in os.walk(jupyter_dir):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)

                # 노트북 상대경로 (jupyter 기준)
                rel_path = os.path.relpath(root, jupyter_dir)

                # 변환 후 저장할 경로
                target_folder = os.path.join(src_dir, rel_path)
                os.makedirs(target_folder, exist_ok=True)

                target_file = os.path.join(target_folder, file[:-6] + ".py")

                # 노트북 파일 읽기
                with open(notebook_path, "r", encoding="utf-8") as f:
                    nb_node = nbformat.read(f, as_version=4)

                # 파이썬 코드로 변환
                python_code, _ = exporter.from_notebook_node(nb_node)

                # py 파일로 저장
                with open(target_file, "w", encoding="utf-8") as f:
                    f.write(python_code)

                print(f"Converted: {notebook_path} -> {target_file}")


if __name__ == "__main__":
    # 스크립트가 위치한 폴더가 jupyter라고 가정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ipynb2py(current_dir)
