# KADIIS
**Anomaly detection in Incomplete Information System via Kernelized Fuzzy-Rough Set**

**abstract**

Anomaly detection is important in knowledge discovery, where fuzzy rough set-based algorithms have been widely applied due to their ability to handle uncertainty and fuzziness in data. However, existing fuzzy rough set-based algorithms exhibit certain limitations when dealing with complex scenarios.
On the one hand, these algorithms typically use linear assumptions to derive fuzzy relations, which fails to effectively handle the complex nonlinearly separable  data, thereby reducing their reliability. 
On the other hand, these algorithms lack mechanisms for handling incomplete data, which limits their applicability and generalizability.
To address these limitations, this paper proposes a kernelized fuzzy rough set theory for incomplete data and applies it to anomaly detection.
This proposed theory employs kernel functions to handle incomplete nonlinearly separable data by defining the incomplete kernelized fuzzy relation alone with its corresponding lower and upper approximations, and futher introduces relevant metrics. 
Building upon this theory, an anomaly detection model is constructed.
First, the indiscernibility degree between objects  is assessed. Subsequently, two indicators across multiple granularities are integrated to compute anomaly factor for each sample. Finally, a corresponging anomaly detection algorithm, named KADIIS, is designed.
Comparative experiments are conducted with nine advanced algorithms on 15 publicly available datasets, and the results demonstrate the effectiveness of KADIIS.

**Usage**

To get the results, simply run the main script:

```
python KADIIS.py
```

Expected Output
Upon successful execution, the script will return the following array of values:

```
[0.5278190500063357, 0.36270988713959884, 0.5162161778278632, 0.22979543350844267, 0.541466147052251, 0.2828692886984754]
```
