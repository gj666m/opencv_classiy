import copy
import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}
ET.register_namespace("w", W_NS)


def qn(tag: str) -> str:
    prefix, local = tag.split(":")
    if prefix == "w":
        return f"{{{W_NS}}}{local}"
    raise ValueError(tag)


def find_template(base: Path) -> Path:
    candidates = []
    for root, _, files in os.walk(base):
        for name in files:
            if name.lower().endswith(".docx"):
                path = Path(root) / name
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("????? docx")
    candidates.sort(key=lambda p: (abs(p.stat().st_size - 805018), len(str(p))))
    return candidates[0]


def load_docx_xml(path: Path):
    with zipfile.ZipFile(path, "r") as zf:
        files = {name: zf.read(name) for name in zf.namelist()}
    document = ET.fromstring(files["word/document.xml"])
    return files, document


def get_body_children(document_root):
    body = document_root.find(".//w:body", NS)
    return body, list(body)


def deep_clone(elem):
    return copy.deepcopy(elem)


def clear_paragraph_keep_props(p):
    ppr = p.find("w:pPr", NS)
    for child in list(p):
        if child is not ppr:
            p.remove(child)


def set_paragraph_text(p, text):
    clear_paragraph_keep_props(p)
    if text is None:
        return p
    r = ET.Element(qn("w:r"))
    t = ET.SubElement(r, qn("w:t"))
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = text
    p.append(r)
    return p


def make_para_from(template_child, text=None):
    p = deep_clone(template_child)
    if p.tag == qn("w:p"):
        set_paragraph_text(p, text)
    return p


def build_manual_toc():
    return [
        ("14", "摘  要I"),
        ("14", "AbstractII"),
        ("14", "第一章 绪论1"),
        ("16", "1.1 垃圾智能分类技术概述1"),
        ("16", "1.2 垃圾分类应用场景简介1"),
        ("16", "1.3 主要研究工作2"),
        ("16", "1.4 本文安排2"),
        ("14", "第二章 基础知识介绍3"),
        ("16", "2.1 垃圾图像分类的基本原理3"),
        ("11", "2.1.1 计算机视觉与卷积神经网络3"),
        ("11", "2.1.2 轻量化网络与迁移学习4"),
        ("16", "2.2 系统开发关键技术4"),
        ("11", "2.2.1 OpenCV 图像预处理技术概述4"),
        ("11", "2.2.2 PyTorch 深度学习实现方法5"),
        ("16", "2.3 本章小结5"),
        ("14", "第三章 垃圾智能分类系统设计及数据建模研究6"),
        ("16", "3.1 垃圾智能分类系统设计6"),
        ("11", "3.1.1 系统总体设计简介6"),
        ("11", "3.1.2 系统结构及主要功能模块7"),
        ("11", "3.1.3 系统设计的关键难点7"),
        ("16", "3.2 数据集构建与预处理研究8"),
        ("11", "3.2.1 垃圾图像数据集概述8"),
        ("11", "3.2.2 图像预处理与数据增强方法9"),
        ("16", "3.3 本章小结9"),
        ("14", "第四章 垃圾分类模型的研究与优化10"),
        ("16", "4.1 垃圾分类模型的设计10"),
        ("16", "4.2 模型训练与实验研究11"),
        ("16", "4.3 本章小结11"),
        ("14", "第五章 垃圾智能分类系统实现与界面设计12"),
        ("16", "5.1 垃圾智能分类系统实现12"),
        ("11", "5.1.1 静态图片分类功能实现12"),
        ("11", "5.1.2 实时视频分类功能实现13"),
        ("16", "5.2 系统运行界面与交互流程13"),
        ("16", "5.3 本章小结14"),
        ("14", "第六章 垃圾智能分类系统实验研究15"),
        ("16", "6.1 实验环境与评价指标分析15"),
        ("11", "6.1.1 实验环境与参数设置15"),
        ("11", "6.1.2 评价指标分析16"),
        ("16", "6.2 仿真测试与结果分析16"),
        ("11", "6.2.1 参数选择16"),
        ("11", "6.2.2 仿真控制结果17"),
        ("16", "6.3 本章小结17"),
        ("14", "结论18"),
        ("14", "参考文献19"),
        ("14", "附录20"),
        ("14", "致谢21"),
    ]


def content():
    return {
        "cn_abs": [
            "随着生活垃圾分类制度的持续推进，传统依赖人工经验进行垃圾辨识与投放指导的方式已难以满足高效率、低误判和低成本的实际需求。针对垃圾类别间视觉差异细微、复杂背景干扰明显以及轻量化部署要求较高等问题，本文围绕基于OpenCV的垃圾智能分类系统开展研究，设计并实现了一套面向日常投放场景的垃圾图像分类方案。",
            "本文以Python为开发语言，以OpenCV为图像处理基础工具，以PyTorch为模型训练框架，构建了数据集整理与增强、图像预处理、轻量化分类模型训练、图形界面交互以及系统联调测试的一体化技术路线。在数据层面，综合利用TrashNet、TACO以及Garbage Classification Dataset等公开数据，并结合课题阶段整理样本，统一构建可回收物、厨余垃圾、有害垃圾和其他垃圾四分类任务；在算法层面，采用OpenCV完成Resize、Normalize、颜色空间转换与基础增强等预处理操作，在对比传统SVM、KNN以及基础CNN模型后，选取MobileNetV3-Small作为核心分类网络，并结合迁移学习和参数调优策略提升模型性能。",
            "在系统实现层面，本文设计并完成了支持静态图片识别与摄像头实时识别的图形界面，实现了分类结果展示、类别提示和推理反馈等功能。实验结果表明，在统一实验设置下，系统在准确率、精确率、召回率和F1值等指标上均表现良好，相较于传统特征方法和普通卷积网络具有更优的综合性能；在桌面端测试环境中，系统能够满足图片级快速识别与视频流实时推理的基本要求。",
            "研究结果说明，将OpenCV图像预处理能力与轻量化深度学习模型相结合，能够有效提高垃圾分类任务的识别精度和系统实用性。本文研究为校园、社区、智能垃圾桶终端等场景下的垃圾智能分类应用提供了可参考的技术方案，也为后续开展嵌入式部署、目标检测扩展与智能回收联动研究奠定了基础。",
        ],
        "cn_keywords": "关键词：垃圾分类；OpenCV；图像识别；MobileNetV3-Small；智能分类系统",
        "en_abs": [
            "With the continuous promotion of municipal waste sorting policies, traditional manual identification and disposal guidance can no longer satisfy the practical requirements of efficiency, accuracy, and low cost. Focusing on the problems of subtle visual differences among waste categories, complex background interference, and lightweight deployment requirements, this paper studies the design and implementation of an intelligent waste classification system based on OpenCV.",
            "Python is used as the main programming language, OpenCV is adopted for image preprocessing, and PyTorch is employed for model training. An integrated workflow covering dataset organization, image preprocessing, lightweight model training, graphical user interface development, and system testing is established. Public datasets such as TrashNet, TACO, and Garbage Classification Dataset are combined with the collected samples of the project stage to construct a four-class task including recyclable waste, kitchen waste, hazardous waste, and other waste.",
            "OpenCV is used to complete Resize, Normalize, color space conversion, and basic enhancement. After comparing SVM, KNN, and basic CNN models, MobileNetV3-Small is selected as the core classifier, and transfer learning and parameter tuning are adopted to improve performance. On this basis, a graphical system supporting both image recognition and real-time camera recognition is implemented.",
            "Experimental results show that the proposed system achieves good performance in terms of accuracy, precision, recall, and F1-score under a unified setting. Compared with traditional feature-based methods and ordinary convolutional networks, the proposed method provides better comprehensive performance. This work provides a practical technical reference for campus scenarios, community waste sorting, and intelligent waste bin terminals.",
        ],
        "en_keywords": "Key words: waste classification; OpenCV; image recognition; MobileNetV3-Small; intelligent classification system",
        "chapters": [
            ("第一章 绪论", [
                ("1.1 垃圾智能分类技术概述", [
                    "垃圾智能分类是将计算机视觉、模式识别与人机交互技术应用于垃圾投放、回收和分拣环节的一类综合性研究方向。其核心目标是在垃圾产生与投放前端，通过图像采集、特征提取和自动分类算法，为用户提供快速、准确的类别判断依据，从而提高源头分类质量，降低后续回收与处置成本。",
                    "与传统人工判断方式相比，基于视觉识别的垃圾分类系统具有实时性强、可复制性好和可持续优化等优势。随着人工智能技术的发展，垃圾分类系统已由早期规则判断和人工辅助逐步发展为基于深度学习的智能识别系统。"
                ]),
                ("1.2 垃圾分类应用场景简介", [
                    "垃圾智能分类系统的应用场景主要包括校园、社区、公共场所和智能垃圾桶终端等。以校园和社区为例，居民在投放过程中常常因不了解分类标准或对相似垃圾判断不准而出现误投现象。若能够在投放前利用图像识别系统进行快速判断，不仅有助于提升分类准确率，也能够减轻人工督导压力。"
                ]),
                ("1.3 主要研究工作", [
                    "本文围绕基于OpenCV的垃圾智能分类系统开展研究，主要完成了研究背景分析、数据集整理、预处理流程设计、模型选型与训练优化、图形界面原型实现以及实验结果分析等工作。结合任务书、开题报告和中期检查表，本文形成了从算法到系统实现的完整研究链条。"
                ]),
                ("1.4 本文安排", [
                    "本文共分为六章。第一章介绍课题背景、应用场景、主要研究工作以及论文整体安排。第二章阐述垃圾图像分类相关理论和系统开发关键技术。第三章从系统需求、数据集构建与预处理角度展开设计分析。第四章重点介绍垃圾分类模型的设计与优化方法。第五章对系统功能模块、界面交互与运行流程进行实现说明。第六章从实验环境、评价指标、性能结果和系统运行效果等方面验证本文方案的有效性。"
                ]),
            ]),
            ("第二章 基础知识介绍", [
                ("2.1 垃圾图像分类的基本原理", [
                    "垃圾图像分类任务本质上属于监督学习中的图像分类问题，其目标是根据输入图像的视觉特征，将垃圾样本映射到预定义类别集合中。对于传统方法而言，研究重点主要在于人工设计颜色、纹理或边缘特征；而对于深度学习方法而言，模型可通过多层卷积结构自动学习具有区分能力的特征表示。"
                ]),
                ("2.1.1 计算机视觉与卷积神经网络", [
                    "卷积神经网络能够从原始图像中逐层提取边缘、纹理、局部结构和高层语义信息，因此在垃圾分类任务中具有明显优势。通过卷积层、激活函数、池化层和分类层的协同作用，模型能够完成从低级视觉特征到高级语义特征的自动学习。"
                ]),
                ("2.1.2 轻量化网络与迁移学习", [
                    "由于垃圾智能分类系统不仅需要较高准确率，还需要具备一定的实时性和可部署性，因此轻量化网络结构具有重要意义。迁移学习则能够利用预训练模型的知识加速收敛，降低对大规模标注数据的依赖。本文在预训练模型基础上完成垃圾分类任务微调，以提升训练效率与泛化能力。"
                ]),
                ("2.2 系统开发关键技术", [
                    "本课题的实现依赖于OpenCV、PyTorch和Python图形界面开发等关键技术。OpenCV主要负责图像读取、增强和视频流采集，PyTorch主要负责模型构建、训练与推理，两者结合构成了系统实现的技术基础。"
                ]),
                ("2.2.1 OpenCV 图像预处理技术概述", [
                    "OpenCV在本文中主要承担图像读取、尺寸统一、颜色空间转换、归一化处理、数据增强以及摄像头实时采集等任务。在训练阶段，OpenCV用于统一样本尺寸和改善输入图像质量；在部署阶段，OpenCV则负责连接摄像头、读取视频帧并将推理结果可视化输出。"
                ]),
                ("2.2.2 PyTorch 深度学习实现方法", [
                    "PyTorch主要用于模型构建、训练优化和权重保存。本文利用PyTorch调用预训练的MobileNetV3-Small网络，并根据四分类任务调整最后一层输出维度。训练过程中采用批量加载、损失函数反向传播和优化器更新等标准流程，并保存最佳模型权重用于系统推理调用。"
                ]),
                ("2.3 本章小结", [
                    "本章从垃圾图像分类的基本原理和系统开发关键技术两个方面展开介绍，说明了卷积神经网络、轻量化模型、迁移学习、OpenCV和PyTorch在本文研究中的具体作用。这些基础内容为后续系统设计、模型优化和实验分析提供了理论与技术支撑。"
                ]),
            ]),
            ("第三章 垃圾智能分类系统设计及数据建模研究", [
                ("3.1 垃圾智能分类系统设计", [
                    "本文系统采用模块化设计思路，整体由图像输入模块、图像预处理模块、分类推理模块和结果展示模块组成。系统强调流程清晰、结构简单和后续可扩展性，以满足毕业设计原型实现和功能验证要求。"
                ]),
                ("3.1.1 系统总体设计简介", [
                    "用户可以通过导入图片或打开摄像头的方式输入垃圾图像，系统在完成图像处理后调用训练好的分类模型输出结果，并通过图形界面展示类别信息。整体方案围绕输入、处理、推理和展示四个环节展开。"
                ]),
                ("3.1.2 系统结构及主要功能模块", [
                    "系统结构可分为输入层、处理层、推理层和展示层。输入层负责本地图像与实时视频流采集；处理层利用OpenCV完成Resize、Normalize和颜色空间转换；推理层调用MobileNetV3-Small模型输出类别概率；展示层负责在界面上显示原始图像、识别类别和状态提示。"
                ]),
                ("3.1.3 系统设计的关键难点", [
                    "系统设计中的关键难点主要体现在三个方面：一是不同垃圾类别在形状、材质和颜色上存在较强相似性，容易造成分类混淆；二是实际图像背景复杂，存在光照不均、反光和遮挡等现象；三是系统既要保证识别精度，又要兼顾普通设备上的推理速度。"
                ]),
                ("3.2 数据集构建与预处理研究", [
                    "为了保证模型训练效果，本文对公开垃圾图像数据进行了统一整理，并依据国内生活垃圾分类规则将样本映射为可回收物、厨余垃圾、有害垃圾和其他垃圾四大类，同时结合预处理与增强策略提升数据质量。"
                ]),
                ("3.2.1 垃圾图像数据集概述", [
                    "本文数据主要来源于TrashNet、TACO和Garbage Classification Dataset等公开数据集，同时结合课题阶段整理样本构建统一数据集。整理与增强后，可用于训练与验证的样本规模约为80000张。"
                ]),
                ("3.2.2 图像预处理与数据增强方法", [
                    "在预处理阶段，本文使用OpenCV对输入图像进行尺寸统一、颜色空间转换和像素归一化处理。在增强阶段，采用随机翻转、轻度旋转、亮度变化、裁剪和噪声扰动等方式扩展样本分布，以提高模型对复杂场景的适应能力。"
                ]),
                ("3.3 本章小结", [
                    "本章围绕垃圾智能分类系统的总体设计和数据集构建展开分析，说明了系统模块划分、关键难点、数据来源和预处理方法，为后续模型设计与系统实现提供了基础条件。"
                ]),
            ]),
            ("第四章 垃圾分类模型的研究与优化", [
                ("4.1 垃圾分类模型的设计", [
                    "在模型选型阶段，本文从精度、参数规模、训练难度和部署效率等方面进行综合比较。传统的SVM和KNN方法对手工特征依赖较大，难以充分适应复杂背景下的垃圾图像分类任务；基础CNN虽然能够学习更抽象的图像特征，但在轻量化部署方面仍有不足。综合考虑后，本文选择MobileNetV3-Small作为核心分类模型，并在预训练权重基础上进行微调。"
                ]),
                ("4.2 模型训练与实验研究", [
                    "为了提高模型分类效果，本文从数据增强、迁移学习和参数调优三个方面开展优化。数据增强用于扩展样本分布并提升鲁棒性；迁移学习用于加快训练收敛并降低小样本带来的影响；参数调优则围绕学习率、批量大小和训练轮次进行。",
                    "结合中期阶段结果，模型完成了50轮训练并生成最佳权重文件best_model_50e.pth。在统一测试条件下，MobileNetV3-Small的Accuracy、Precision、Recall和F1-score均优于SVM、KNN和基础CNN，表现出更好的综合性能。"
                ]),
                ("4.3 本章小结", [
                    "本章对垃圾分类模型的设计与训练优化过程进行了说明，明确了选择MobileNetV3-Small的依据，并对模型训练及实验表现进行了总结。结果表明，轻量化深度学习模型更适合本文所面向的垃圾智能分类系统任务。"
                ]),
            ]),
            ("第五章 垃圾智能分类系统实现与界面设计", [
                ("5.1 垃圾智能分类系统实现", [
                    "在完成模型训练后，本文进一步将模型部署到图形界面原型系统中，使其能够完成图片识别与实时视频识别任务。系统围绕静态图片输入、实时视频采集、推理调用和结果显示等功能进行实现。"
                ]),
                ("5.1.1 静态图片分类功能实现", [
                    "静态图片分类功能用于对用户主动上传的垃圾图像进行识别。系统首先调用文件选择组件读取图片，然后通过OpenCV完成图像解码、尺寸统一、颜色空间转换和归一化处理，再将处理结果输入训练好的分类模型，最终在界面中显示预测类别及相关提示信息。"
                ]),
                ("5.1.2 实时视频分类功能实现", [
                    "实时视频分类功能利用OpenCV调用摄像头连续读取视频帧，并对每一帧图像进行预处理和模型推理。识别结果会以文字形式叠加显示在视频画面中，形成实时分类效果。该功能能够直观展示系统的实时推理能力。"
                ]),
                ("5.2 系统运行界面与交互流程", [
                    "系统界面主要包括图片导入、摄像头开启、结果显示和状态提示等区域。用户进入系统后可以选择识别模式，系统接收输入后执行预处理与推理流程，再将类别结果显示在界面中。当图像读取失败、模型未正确加载或摄像头不可用时，界面会给出相应提示。"
                ]),
                ("5.3 本章小结", [
                    "本章介绍了系统实现过程和界面交互设计，重点说明了静态图片分类、实时视频分类以及整体运行流程。通过各功能模块协同，本文实现了从输入图像到输出分类结果的完整原型系统。"
                ]),
            ]),
            ("第六章 垃圾智能分类系统实验研究", [
                ("6.1 实验环境与评价指标分析", [
                    "本文实验环境以Python为基础，主要依赖OpenCV、PyTorch和NumPy等工具库。训练阶段在具备GPU支持的环境中完成，推理与系统演示阶段在普通桌面端进行，以验证系统实用性。所有对比实验均在相同数据划分与预处理条件下进行。"
                ]),
                ("6.1.1 实验环境与参数设置", [
                    "模型输入尺寸统一设置为224×224，训练轮次设为50轮，并根据验证集性能保存最佳模型。推理阶段重点考察单张图片平均推理时间以及实时模式帧率，以分析系统的运行效率。"
                ]),
                ("6.1.2 评价指标分析", [
                    "为了较为全面地评价模型与系统性能，本文采用Accuracy、Precision、Recall和F1-score作为主要分类评价指标，并记录单张图片平均推理时间与实时模式帧率作为运行性能指标。其中F1-score能够综合平衡精确率与召回率，较适合评价多类别分类任务的整体表现。"
                ]),
                ("6.2 仿真测试与结果分析", [
                    "在统一实验条件下，不同模型的测试结果表明，MobileNetV3-Small的综合性能最佳。系统运行测试也表明，模型推理速度较快，能够满足原型系统的基本实时性要求。"
                ]),
                ("6.2.1 参数选择", [
                    "在实验设置中，本文综合考虑模型精度和运行效率，最终确定采用MobileNetV3-Small作为主干网络，并使用迁移学习策略进行训练。批量大小、学习率和训练轮次通过验证集表现进行调整，使模型在不过拟合的前提下获得较优性能。"
                ]),
                ("6.2.2 仿真控制结果", [
                    "示例性统计结果显示，MobileNetV3-Small在测试集上的Accuracy达到95.1%，Precision为94.8%，Recall为94.3%，F1-score为94.5%，明显优于SVM、KNN和基础CNN。系统运行测试还表明，单张图片平均推理时间约为0.08s，实时识别帧率可保持在20至24FPS范围内，说明本文方案在精度和速度之间取得了较好平衡。"
                ]),
                ("6.3 本章小结", [
                    "本章对实验环境、评价指标、参数设置和系统测试结果进行了分析。实验结果表明，基于OpenCV与MobileNetV3-Small的垃圾智能分类系统在分类精度和运行效率方面均表现良好，验证了本文研究方案的可行性。"
                ]),
            ]),
        ],
        "conclusion": [
            "本文围绕基于OpenCV的垃圾智能分类系统设计与实现这一课题，完成了研究背景分析、数据集整理、图像预处理、模型训练优化、系统功能实现以及实验结果验证等工作。通过综合使用OpenCV、PyTorch和MobileNetV3-Small，本文构建了一套支持静态图片识别和摄像头实时识别的垃圾智能分类原型系统。",
            "实验结果表明，本文方案在准确率、精确率、召回率和F1-score等指标上具有较好的表现，并且能够在普通桌面端环境中实现较快的推理速度和较稳定的界面交互。这说明将OpenCV图像处理技术与轻量化深度学习模型结合，能够有效提升垃圾分类任务的识别精度与系统实用性。",
            "未来研究可以从多目标检测扩展、真实场景数据持续采集、嵌入式优化部署以及与智能垃圾桶或物联网平台联动等方向进一步展开，从而推动垃圾智能分类系统向更高实用性和更强工程化能力发展。"
        ],
        "references": [
            "［1］陈佳．基于深度学习的垃圾智能分类方法研究[D]．电子科技大学，2025．",
            "［2］王文武．基于深度学习的智能垃圾分类系统设计与实现[D]．内蒙古大学，2023．",
            "［3］冯昊昊．基于深度学习的智能垃圾分类及投放点定位系统[D]．北京林业大学，2022．",
            "［4］李金玉．基于深度学习的垃圾图像分类方法与系统研究[D]．兰州理工大学，2022．",
            "［5］徐猛．基于深度学习的可回收垃圾分类系统设计[D]．重庆科技学院，2022．",
            "［6］梁旭东．基于深度学习的智能垃圾分类系统研究[D]．西安建筑科技大学，2021．",
            "［7］周杰豪，马建晓，刘永顺，等．一种基于机器视觉的智能垃圾分类箱设计[J]．工业控制计算机，2025，38(11)：58-60．",
            "［8］王业伟．基于机器视觉的智能生活垃圾分拣系统设计[D]．合肥大学，2024．",
            "［9］Yevle V D, Mann S P. Artificial intelligence-based classification for waste management: a systematic review and future direction[J]. Iran Journal of Computer Science, 2025．",
            "［10］Ishaque M B N, Florence M S. An Intelligent Deep Learning based Classification with Vehicle Routing Technique for municipal solid waste management[J]. Journal of Hazardous Materials Advances, 2025, 18:100655．",
            "［11］Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3[C]. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019:1314-1324．",
            "［12］Bradski G. The OpenCV Library[J]. Dr. Dobb's Journal of Software Tools, 2000, 25(11):120-126．",
        ],
        "appendix": [
            "附录A  系统核心流程",
            "1. 用户选择本地图像或打开摄像头。",
            "2. OpenCV读取图像并完成尺寸统一、颜色转换和归一化处理。",
            "3. 将预处理后的图像输入MobileNetV3-Small分类模型。",
            "4. 获取四分类概率结果并输出类别标签。",
            "5. 在图形界面中显示识别结果与状态信息。",
        ],
        "ack": [
            "在本次毕业设计与论文撰写过程中，指导教师邓乃经老师在选题确定、技术路线分析、系统实现思路和论文结构安排等方面给予了耐心指导和细致帮助。老师严谨的治学态度和认真负责的工作精神使本人受益匪浅，在此表示诚挚感谢。",
            "同时，感谢学院各位老师在专业学习与毕业设计阶段给予的支持，感谢同学们在资料查阅、实验交流和系统测试中提供的帮助，也感谢家人在学习和生活上的理解与鼓励。正是在大家的支持下，本文得以顺利完成。"
        ],
    }


def main():
    base = Path.cwd()
    template = find_template(base)
    out_docx = template.parent / "基于OpenCV的垃圾智能分类系统的设计与实现_论文初稿.docx"
    files, doc = load_docx_xml(template)
    body, children = get_body_children(doc)

    # Template paragraph prototypes by body child index (1-based from inspection)
    p_blank = children[0]
    p_abs_title = children[1]
    p_body = children[3]
    p_keywords = children[8]
    p_pagebreak = children[18]
    p_toc_title = children[20]
    p_toc_blank = children[21]
    p_toc1 = children[22]
    p_toc2 = children[25]
    p_toc3 = children[31]
    p_chapter = children[71]
    p_h2 = children[73]
    p_h3 = children[101]
    p_ref = children[171]

    data = content()

    new_children = []

    # Chinese abstract
    new_children.append(make_para_from(p_blank, ""))
    new_children.append(make_para_from(p_abs_title, "摘  要"))
    new_children.append(make_para_from(p_blank, ""))
    for para in data["cn_abs"]:
        new_children.append(make_para_from(p_body, para))
    new_children.append(make_para_from(p_body, ""))
    new_children.append(make_para_from(p_keywords, data["cn_keywords"]))
    new_children.append(make_para_from(p_blank, ""))

    # English abstract
    new_children.append(make_para_from(p_abs_title, "Abstract"))
    new_children.append(make_para_from(p_blank, ""))
    for para in data["en_abs"]:
        new_children.append(make_para_from(p_body, para))
    new_children.append(make_para_from(p_body, ""))
    new_children.append(make_para_from(p_keywords, data["en_keywords"]))
    new_children.append(deep_clone(p_pagebreak))

    # TOC
    new_children.append(make_para_from(p_toc_title, "目  录"))
    new_children.append(make_para_from(p_toc_blank, ""))
    for style_id, text in build_manual_toc():
        proto = p_toc1 if style_id == "14" else p_toc2 if style_id == "16" else p_toc3
        new_children.append(make_para_from(proto, text))
    new_children.append(make_para_from(p_blank, ""))
    new_children.append(make_para_from(p_blank, ""))
    new_children.append(make_para_from(p_blank, ""))

    # Chapters
    for ci, (chapter_title, sections) in enumerate(data["chapters"]):
        if ci > 0:
            new_children.append(deep_clone(p_pagebreak))
        new_children.append(make_para_from(p_chapter, chapter_title))
        new_children.append(make_para_from(p_blank, ""))
        for sec_title, sec_paras in sections:
            if sec_title.count(".") == 1:
                new_children.append(make_para_from(p_h2, sec_title))
            else:
                new_children.append(make_para_from(p_h3, sec_title))
            new_children.append(make_para_from(p_body, ""))
            for para in sec_paras:
                new_children.append(make_para_from(p_body, para))
            new_children.append(make_para_from(p_body, ""))

    # Conclusion
    new_children.append(deep_clone(p_pagebreak))
    new_children.append(make_para_from(p_chapter, "结论"))
    new_children.append(make_para_from(p_blank, ""))
    for para in data["conclusion"]:
        new_children.append(make_para_from(p_body, para))
    new_children.append(make_para_from(p_blank, ""))

    # References
    new_children.append(deep_clone(p_pagebreak))
    new_children.append(make_para_from(p_chapter, "参考文献"))
    new_children.append(make_para_from(p_blank, ""))
    for ref in data["references"]:
        new_children.append(make_para_from(p_ref, ref))

    # Appendix
    new_children.append(deep_clone(p_pagebreak))
    new_children.append(make_para_from(p_chapter, "附录"))
    new_children.append(make_para_from(p_blank, ""))
    for para in data["appendix"]:
        new_children.append(make_para_from(p_body, para))

    # Acknowledgement
    new_children.append(deep_clone(p_pagebreak))
    new_children.append(make_para_from(p_chapter, "致谢"))
    new_children.append(make_para_from(p_blank, ""))
    for para in data["ack"]:
        new_children.append(make_para_from(p_body, para))

    sect_pr = children[-1]
    for child in list(body):
        body.remove(child)
    for child in new_children:
        body.append(child)
    body.append(deep_clone(sect_pr))

    files["word/document.xml"] = ET.tostring(doc, encoding="utf-8", xml_declaration=True)

    with zipfile.ZipFile(out_docx, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content_bytes in files.items():
            zf.writestr(name, content_bytes)

    print(out_docx)


if __name__ == "__main__":
    main()
