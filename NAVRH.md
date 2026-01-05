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
1. HAILO TECHNOLOGIES LTD. Automatic License Plate Recognition with Hailo Processors [online]. Hailo blog. 2024 [cit. 2025-12-04]. Dostupné z: https://hailo.ai/ja/blog/automatic-license-plate-recognition-with-hailo-processors/
2. HYNDMAN, Rob J. a George ATHANASOPOULOS. Forecasting: principles and practice. Second edition. [Melbourne]: OTexts, 2018. ISBN 978-0-9875071-1-2.
3. RASPBERRY PI FOUNDATION. AI Kit and AI HAT+ software [online]. [cit. 2025-12-04]. Dostupné z: https://www.raspberrypi.com/documentation/computers/ai.html
4. REDMON, Joseph, Santosh DIVVALA, Ross GIRSHICK a Ali FARHADI. You Only Look Once: Unified, Real-Time Object Detection. ArXiv.org [online]. 2015, 2016-05-09, 10 [cit. 2025-11-03]. Dostupné z:  https://arxiv.org/abs/1506.02640
5. SZELISKI, Richard. Computer vision: algorithms and applications. Second edition. Cham, Switzerland: Springer. ISBN 978-3-030-34372-9.
6. ZHOU, Wei, Li YANG, Lei ZHAO, Runyu ZHANG a ostatní. Vision Technologies with Applications in Traffic Surveillance Systems: A Holistic Survey. ArXiv.org [online]. 2025-06-28, 2024, 25 [cit. 2025-12-04]. Dostupné z:  https://arxiv.org/abs/2412.00348