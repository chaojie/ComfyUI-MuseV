import io
from typing import List, Dict, Tuple, BinaryIO
from copy import deepcopy

import pandas as pd
from PIL import Image
from xlsxwriter.utility import xl_col_to_name

def read_image_data(path: str, target_image_height: int) -> Tuple[BinaryIO, int, int]:
    """读取图像，按照目标高做resize处理，并转化成二进制格式，返回最终图像的宽和高

    Args:
        path (str): 图像路径
        target_image_height (int): 目标高

    Returns:
        Tuple[BinaryIO, int, int]: 图像二进制格式，返回最终图像的宽和高
    """
    image = Image.open(path)
    image_width_in_excel = int(image.width / (image.height / target_image_height))
    image = image.resize(size=(image_width_in_excel, target_image_height))
    image_byte = io.BytesIO()
    image.save(image_byte, format="png")
    return image_byte, image.width, image.height


def set_text_column_dynamic_width(worksheet, df, format, default_width=50):
    """将df在excel workshhet中的列按照实际内容长度设置列宽以及文本格式

    Args:
        worksheet (_type_): 待处理的excel worksheet
        df (_type_): worksheet中原来对应的DataFrame格式
        format (_type_): 对应列的文本格式
        default_width (int, optional): 默认目标宽度. Defaults to 50.
    """
    for column in df:
        column_width = max(df[column].astype(str).map(len).max(), len(column))
        col_idx = df.columns.get_loc(column)
        width = min(column_width, default_width)
        worksheet.set_column(col_idx, col_idx, width, format)


def convert_tasks2clean(tasks):
    tasks = [{"prompt": task["prompt"]} for task in tasks]
    return tasks


def split_tasks_by_images_lst(tasks, save_images_path_key: str = "save_images_path"):
    new_tasks = []
    for task in tasks:
        for image_path in task[save_images_path_key]:
            new_task = deepcopy(task)
            new_task[save_images_path_key] = image_path
            new_tasks.append(new_task)
    return new_tasks


def save_texts_images_2_csv(tasks: List[Dict], save_path: str):
    """存储相关结果为csv表格

    tasks (List[Dict]): 待转换的字典列表
    save_path (str): 表格存储路径
    """
    df = pd.DataFrame(tasks)
    df.to_csv(save_path, encoding="utf_8_sig", index=False)


def add_multi_data_validation(workbook, worksheet, validates, validate_idxs, n_rows):
    for i, validate in enumerate(validates):
        validate_idx = validate_idxs[i]
        worksheet = add_data_validation(
            workbook=workbook,
            worksheet=worksheet,
            col=validate_idx,
            head=validate["col_name"],
            candidates=validate["candidates"],
            colors=validate["colors"],
            n_rows=n_rows,
        )
    return worksheet


def add_data_validation(
    workbook, worksheet, col: int, head, candidates, n_rows, colors
):
    col = xl_col_to_name(col)
    # Adding the header and Datavalidation list
    worksheet.write('{}1'.format(col), head)
    colors_fmt = [workbook.add_format({'bg_color': color}) for color in colors]
    for row in range(n_rows):
        cell_idx = '{}{}'.format(col, row + 2)
        worksheet.data_validation(cell_idx, {'validate': 'list', 'source': candidates})
        for i_c in range(len(candidates)):
            worksheet.conditional_format(
                cell_idx,
                {
                    'type': 'formula',
                    'criteria': '=${}=\"{}\"'.format(cell_idx, candidates[i_c]),
                    'format': colors_fmt[i_c],
                },
            )
    return worksheet


def insert_cell_image(
    worksheet,
    row,
    col,
    image_path,
    image_height_in_table,
    text_format,
    row_ratio,
    col_ratio,
):
    image_byte, new_image_width, new_image_height = read_image_data(
        image_path, target_image_height=image_height_in_table
    )
    # TODO：现在的图像列并不是预期内的和图像等宽，而是宽了很多
    worksheet.set_column(
        col,
        col,
        int(new_image_width / col_ratio),
    )
    worksheet.insert_image(
        row,
        col,
        image_path,
        {"image_data": image_byte},
    )
    worksheet.set_row(row, int(new_image_height / row_ratio), text_format)
    return worksheet


def save_texts_images_2_excel(
    tasks: List[Dict],
    save_path: str,
    image_height_in_table: int = 120,
    row_ratio: float = 1.3,
    col_ratio: float = 5,
    validates: List = None,
):
    """将任务列表和生成的图像统一存储在表格中，方便观看对比实验结果。

    Args:
        tasks (List[Dict]): 待转换的字典列表
        save_path (str): 表格存储路径
        image_height_in_table (int, optional): 表格中缩略图的高. Defaults to 120.
        row_ratio (float, optional): excel的单元格宽高和实际图像边长需要做比例转换. Defaults to 1.2.
        col_ratio (float, optional): excel的单元格宽高和实际图像边长需要做比例转换. Defaults to 7.5.
        need_add_checker_column (bool, optional): 是否新增一列用于审核检查状态. Defaults to False.
    """
    df = pd.DataFrame(tasks)
    # 先找到需要插入图像的列，插入图像列
    keys_with_image = [
        k for k in tasks[0].keys() if "images_path" in k and k != "save_images_path"
    ]
    high_priority_col_idx = 0
    # 默认save_images_path是生成图像，放在后面
    if "save_images_path" in tasks[0]:
        keys_with_image.append("save_images_path")
    for img_key in keys_with_image:
        maxlen_img_key_value = max(
            [
                len(task[img_key]) if isinstance(task[img_key], list) else 1
                for task in tasks
            ]
        )
        for i in range(maxlen_img_key_value):
            column = "{}_{}".format(img_key, i)
            if column not in df.columns:
                df.insert(
                    loc=high_priority_col_idx,
                    column=column,
                    value="",
                )
                high_priority_col_idx += 1

    validate_start_idx = high_priority_col_idx
    if validates is not None:
        for i, validate in enumerate(validates):
            if validate["col_name"] not in df.columns:
                col_idx = validate_start_idx + i
                df.insert(loc=col_idx, column=validate["col_name"], value="")
        validate_idxs = range(validate_start_idx, validate_start_idx + len(validates))
    writer = pd.ExcelWriter(save_path, engine="xlsxwriter")
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name="Sheet1", index=False)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    n_col = len(df.columns)
    n_row = len(df)
    text_format = workbook.add_format({"text_wrap": True})
    set_text_column_dynamic_width(worksheet=worksheet, df=df, format=text_format)
    # Insert an image.
    for row in range(len(df)):
        task = tasks[row]
        cell_col = 0
        for im_idx, img_key in enumerate(keys_with_image):
            images_path = task[img_key]
            if not isinstance(images_path, list):
                cell_row = 1 + row
                worksheet.write(cell_row, cell_col, img_key)
                if len(images_path) == 0:
                    continue
                worksheet = insert_cell_image(
                    worksheet=worksheet,
                    row=cell_row,
                    col=cell_col,
                    image_path=images_path,
                    image_height_in_table=image_height_in_table,
                    text_format=text_format,
                    row_ratio=row_ratio,
                    col_ratio=col_ratio,
                )
                cell_col += 1
            else:
                for i, image_path in enumerate(images_path):
                    worksheet = insert_cell_image(
                        worksheet=worksheet,
                        row=cell_row,
                        col=cell_col,
                        image_path=image_path,
                        image_height_in_table=image_height_in_table,
                        text_format=text_format,
                        row_ratio=row_ratio,
                        col_ratio=col_ratio,
                    )
                    cell_col += 1
    if validates is not None:
        worksheet = add_multi_data_validation(
            workbook,
            worksheet,
            validates=validates,
            validate_idxs=validate_idxs,
            n_rows=len(df),
        )
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
