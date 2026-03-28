# Bachelor Thesis Project Proposal

## Project Titles
* **Czech:** Systém pro monitorování a analýzu dopravy pomocí počítačového vidění
* **English:** System for Traffic Monitoring and Analysis using Computer Vision

## Abstract
Collecting traffic data—such as traffic density, vehicle types, and pedestrian or cyclist movement—is crucial for the planning and optimization of transportation infrastructure. Traditional methods, such as manual counting, are time-consuming and prone to human error. A potential solution lies in the deployment of an automated system capable of real-time video signal analysis from traffic locations to detect and classify various road users (cars, pedestrians, cyclists) and collect relevant data.

The goal of this bachelor's thesis is to design, implement, and test a system for the automated counting and analysis of vehicle and pedestrian flow in a defined location. The system will be built on the **Raspberry Pi** platform equipped with an **AI module**. It will be capable of real-time image processing, specifically analyzing video streams to detect and classify objects, collect data, and visualize the results in a user-friendly application.

## Specific Objectives
* **Literature Review:** Review of methods for real-time object detection, tracking, and classification (e.g., YOLO, SSD, DeepSORT).
* **Market Analysis:** Analysis of existing solutions for traffic monitoring.
* **Architecture Design:** Design of the system architecture, including the camera module, processing unit, and software application.
* **Dataset Preparation:** Data collection and preparation using existing public datasets (e.g., COCO, BDD100K) or creating a custom small dataset for model training or fine-tuning.
* **Model Implementation:** Implementation of the detection model.
* **Software Development:** Implementation of an application providing:
    * Real-time video signal analysis.
    * Detection of crossings/passages over a defined line or area.
    * Structured data storage (object type, timestamp, direction) in a database.
    * Data visualization (charts and statistics) via a web interface.
* **Evaluation:** Testing in a real-world environment and assessing the accuracy compared to manual observation.

## Specific Objectives
* **Literature Review:** Review of methods for real-time object detection, tracking, and classification (e.g., YOLO, SSD, DeepSORT).
* **Market Analysis:** Analysis of existing solutions for traffic monitoring.
* **Architecture Design:** Design of the system architecture, including the camera module, processing unit, and software application.
* **Dataset Preparation:** Data collection and preparation using existing public datasets (e.g., COCO, BDD100K) or creating a custom small dataset for model training or fine-tuning.
* **Model Implementation:** Implementation of the detection model.
* **Software Development:** Implementation of an application providing:
    * Real-time video signal analysis.
    * Detection of crossings/passages over a defined line or area.
    * Structured data storage (object type, timestamp, direction) in a database.
    * Data visualization (charts and statistics) via a web interface.
* **Evaluation:** Testing in a real-world environment and assessing the accuracy compared to manual observation.

### Bibliography
1. EVERINGHAM, Mark, Luc VAN GOOL, Christopher K. I. WILLIAMS, John WINN a Andrew ZISSERMAN. The Pascal Visual Object Classes (VOC) Challenge. International Journal of Computer Vision [online]. Springer Science and Business Media, 2009, 2009-9-9, 88(2), 303-338 [cit. 2026-02-13]. ISSN 0920-5691. Dostupné z: doi:10.1007/s11263-009-0275-4
2. LI, En, Liekang ZENG, Zhi ZHOU a Xu CHEN. Edge AI: On-Demand Accelerating Deep Neural Network Inference via Edge Computing. IEEE Transactions on Wireless Communications [online]. 2020, 19(1), s. 447-457 [cit. 2026-02-14]. Dostupné z: doi:10.1109/TWC.2019.2946140
3. REDMON, Joseph, Santosh DIVVALA, Ross GIRSHICK a Ali FARHADI. You Only Look Once: Unified, Real-Time Object Detection. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) [online]. IEEE, 2016, 779-788 [cit. 2026-02-13]. Dostupné z: doi:10.1109/cvpr.2016.91
4. SZELISKI, Richard. Computer vision: algorithms and applications. Second edition. Cham, Switzerland: Springer, [2022]. Texts in computer science. ISBN 978-3-030-34372-9.
5. WOJKE, Nicolai, Alex BEWLEY a Dietrich PAULUS. Simple online and realtime tracking with a deep association metric. 2017 IEEE International Conference on Image Processing (ICIP) [online]. IEEE, 2017, 2017, 3645-3649 [cit. 2026-02-13]. Dostupné z: doi:10.1109/icip.2017.8296962
6. ZHOU, Wei, Li YANG, Lei ZHAO, Runyu ZHANG, Yifan CUI, Hongpu HUANG, Kun QIE a Chen WANG. Vision Technologies with Applications in Traffic Surveillance Systems: A Holistic Survey. ACM Computing Surveys [online]. Association for Computing Machinery (ACM), 2025, 2025-9-9, 58(3), 1-47 [cit. 2026-02-13]. ISSN 0360-0300. Dostupné z: doi:10.1145/3760525
