# This file contains necesasry functions to run the PINN algorithm


# import libraries
#import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Multiply
#, Add, Subtract
from bokeh.layouts import column, row
from bokeh.io import output_notebook, show,export_png, save
from bokeh.io import output_file
from bokeh.plotting import figure
from bokeh import palettes
##导出为矢量图
from bokeh.io import export_svgs
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
# 确保在文件开头导入这些库
import os
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from bokeh.io import export_svgs
import random

# functions
#mse = tf.keras.losses.MeanSquaredError()

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
def set_global_determinism(seed=0):
    set_seeds(seed=seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # 禁用 GPU，强迫 TensorFlow 在 CPU 上执行（确定性）
    # tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# figure font adjustments
def figure_settings(fig,label_font_size='12pt',legend_fix=True):
    fig.yaxis.axis_label_text_font_size  = '16pt'
    fig.xaxis.major_label_text_font_size = label_font_size
    fig.yaxis.major_label_text_font_size = label_font_size
    fig.yaxis.major_label_text_font_style = "bold"
    fig.xaxis.major_label_text_font_style = "bold"
    fig.xaxis.axis_label_text_font_size  = '16pt'
    fig.axis.axis_label_text_font_style = 'bold'
    
    # fig.legend.location = 'bottom_right'
    if legend_fix:
        fig.legend.label_text_font_size = '14pt'
        fig.legend.title_text_font = 'Arial'
    return fig

def save_svg_and_pdf(bokeh_layout, name):
    svg_path = f"fig/{name}.svg"
    pdf_path = f"fig/{name}.pdf"

    # 1. 设置 Firefox 浏览器的位置 (请根据你电脑的实际情况检查这个路径是否存在)
    firefox_binary_path = r"E:\programs-app\anaconda3\envs\py3.9pinncbp\Library\bin\firefox.exe"

    # 2. 设置 Geckodriver 的位置 (因为它就在你的项目文件夹下，直接写文件名即可)
    driver_path = r"E:\programs-app\anaconda3\envs\py3.9pinncbp\Library\bin\geckodriver.exe"

    # --- 配置 WebDriver ---
    options = Options()
    options.binary_location = firefox_binary_path  # 告诉它浏览器在哪
    options.add_argument("--headless")  # 无头模式（后台运行，不弹窗）

    service = Service(executable_path=driver_path)  # 告诉它驱动在哪

    driver = None
    try:
        # 初始化浏览器
        driver = webdriver.Firefox(service=service, options=options)

        # 导出 SVG (传入配置好的 driver)
        export_svgs(bokeh_layout, filename=svg_path, webdriver=driver)

        print(f"Saved SVG: {svg_path}")

        # SVG 转 PDF
        drawing = svg2rlg(svg_path)
        renderPDF.drawToFile(drawing, pdf_path)
        print(f"Saved PDF: {pdf_path}")

    except Exception as e:
        print(f"导出失败: {e}")
        # 如果报错，打印出它到底在找哪里的路径，方便调试
        if not os.path.exists(firefox_binary_path):
            print(f"!!! 关键错误: 找不到 Firefox 浏览器，请检查路径: {firefox_binary_path}")
        if not os.path.exists(driver_path):
            print(f"!!! 关键错误: 找不到 geckodriver，请确认它在项目根目录下: {os.path.abspath(driver_path)}")

    finally:
        if driver:
            driver.quit()