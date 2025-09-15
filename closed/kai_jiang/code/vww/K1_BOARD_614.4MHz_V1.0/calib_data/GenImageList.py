import os

def generate_txt(directory, output_file):
    with open(output_file, 'w') as out_file:
        base_dir = os.path.dirname(os.path.abspath(output_file))
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # 只处理常见的图片文件类型
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, base_dir)
                    out_file.write(relative_path + '\n')

if __name__ == "__main__":
    input_directory = input("请输入包含图片文件的目录: ")
    output_txt = "output.txt"  # 你可以根据需要修改输出文件名
    generate_txt(input_directory, output_txt)