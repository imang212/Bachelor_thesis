# Návrh zadání závěrečné práce (Bakalářská práce)
## Název tématu v češtině:
Systém pro monitorování a analýzu dopravy pomocí počítačového vidění

## Název tématu v angličtině:
System for Traffic Monitoring and Analysis using Computer Vision

### Anotace tématu
Sběr dat o dopravě, jako je hustota provozu, typy vozidel nebo pohyb chodců a cyklistů, je klíčový pro plánování a optimalizaci dopravní infrastruktury. Tradiční metody například manuální sčítání jsou časově náročné a náchylné k chybám. Možným řešením je využití automatizovaného systému, který dokáže v reálném čase analyzovat video signál z dopravní lokality, detekovat a klasifikovat různé účastníky provozu (auta, chodce, cyklisty) a sbírat o nich data.

Cílem této bakalářské práce je navrhnout, implementovat a otestovat systém pro automatické sčítání a analýzu průjezdů vozidel a průchodů osob v definované lokalitě. Systém bude postaven na platformě Raspberry Pi s AI modulem a bude schopen v reálném čase zpracovávat obraz tj. analyzovat video signál z dopravní lokality, detekovat a klasifikovat objekty (vozidla, lidé, cyklisté), sbírat data o nich a vizualizovat je v přehledné aplikaci.

Dílčí cíle:

*  přehled metod pro detekci, sledování a klasifikaci objektů v reálném čase (např. YOLO, SSD, DeepSORT),
*  analýza dostupných řešení pro dopravní monitoring,
*  návrh architektury systému zahrnující kamerový modul, zpracovatelskou jednotku a softwarovou aplikaci,
*  sběr a příprava dat s využitím existujících veřejných datových sad (např. COCO, BDD100K) nebo vytvoření vlastní malé datové sady pro trénování nebo fine-tuning modelu,
*  implementace modelu,
*   vývoj softwaru: Implementace aplikace, která bude poskytovat tyto funkce
    *   analýza videosignálu z kamery v reálném čase,
    *   detekce průjezdů a průchodů objektů definovanou čarou nebo oblastí,
    *   uložení dat (typ objektu, čas, směr) do databáze nebo souboru,
    *   vizualizace dat v podobě grafů a statistik (např. v jednoduchém webovém rozhraní),
*   testování a evaluace v reálném prostředí, zhodnocení přesnosti detekce a sčítání ve srovnání s manuálním pozorováním.

Výstupem práce bude funkční prototyp zařízení a softwarová aplikace pro monitorování dopravy. Součástí výstupu bude i dokumentace popisující návrh, implementace a vyhodnocení přesnosti systému.

Osnova:

Teoretická část
1.   metody počítačového vidění pro detekci a sledování objektů
2.   přehled systémů pro analýzu a zpracování dat z dopravy
3.   AI na vestavěných systémech (Edge AI)
4.   metodika testování a srovnání s referenčními daty

Praktická část
1.  celková architektura řešení
2.  výběr hardwaru (Raspberry Pi, kamera, AI akcelerátor)
3.  návrh softwarových komponent (detekční modul, databáze, vizualizační rozhraní)
4.  příprava prostředí a dat
5.  trénování a nasazení detekčního modelu
6.  vývoj aplikace pro zpracování videa a sběr dat
7.  vytvoření vizualizačního rozhraní
8.  analýza přesnosti a výkonu systému
9.  Prezentace a interpretace výsledků

### Literatura:

1. EVERINGHAM, Mark, Luc VAN GOOL, Christopher K. I. WILLIAMS, John WINN a Andrew ZISSERMAN. The Pascal Visual Object Classes (VOC) Challenge. International Journal of Computer Vision [online]. Springer Science and Business Media, 2009, 2009-9-9, 88(2), 303-338 [cit. 2026-02-13]. ISSN 0920-5691. Dostupné z: doi:10.1007/s11263-009-0275-4
2. LI, En, Liekang ZENG, Zhi ZHOU a Xu CHEN. Edge AI: On-Demand Accelerating Deep Neural Network Inference via Edge Computing. IEEE Transactions on Wireless Communications [online]. 2020, 19(1), s. 447-457 [cit. 2026-02-14]. Dostupné z: doi:10.1109/TWC.2019.2946140
3. REDMON, Joseph, Santosh DIVVALA, Ross GIRSHICK a Ali FARHADI. You Only Look Once: Unified, Real-Time Object Detection. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR) [online]. IEEE, 2016, 779-788 [cit. 2026-02-13]. Dostupné z: doi:10.1109/cvpr.2016.91
4. SZELISKI, Richard. Computer vision: algorithms and applications. Second edition. Cham, Switzerland: Springer, [2022]. Texts in computer science. ISBN 978-3-030-34372-9.
5. WOJKE, Nicolai, Alex BEWLEY a Dietrich PAULUS. Simple online and realtime tracking with a deep association metric. 2017 IEEE International Conference on Image Processing (ICIP) [online]. IEEE, 2017, 2017, 3645-3649 [cit. 2026-02-13]. Dostupné z: doi:10.1109/icip.2017.8296962
6. ZHOU, Wei, Li YANG, Lei ZHAO, Runyu ZHANG, Yifan CUI, Hongpu HUANG, Kun QIE a Chen WANG. Vision Technologies with Applications in Traffic Surveillance Systems: A Holistic Survey. ACM Computing Surveys [online]. Association for Computing Machinery (ACM), 2025, 2025-9-9, 58(3), 1-47 [cit. 2026-02-13]. ISSN 0360-0300. Dostupné z: doi:10.1145/3760525
