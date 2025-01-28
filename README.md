# **Retinex-Guided Diffusion-Based Low-Light Image Enhancement**

## **Overview**
This repository focuses on **low-light image enhancement (LLIE)** using **Retinex theory** and advanced methods like **diffusion models**. The project includes a comprehensive survey of LLIE algorithms and implementation of the **Retinex-Net** model with perceptual loss to improve image quality under low-light conditions.

---

## **Motivation**
This project marks an important step in exploring **image enhancement techniques** for low-light scenarios. It aims to:
- Address challenges in photography and imaging systems under poor lighting.
- Study and apply **Retinex theory** for image decomposition into illumination and reflectance components.
- Integrate **diffusion models** to further enhance illumination for better results.

---

## **Methodology**

### **1️⃣ Literature Survey**
A comprehensive study was conducted on various LLIE methods, focusing on Retinex-based approaches:

#### **Retinex-Net**  
**Retinex-Net** proposes a data-driven approach to Retinex decomposition for low-light image enhancement. It consists of two main components:  
1. **Decom-Net**: Decomposes low-light images into **illumination maps** and **reflectance maps** by learning the inherent structure of paired low/normal light images.  
2. **Enhance-Net**: Enhances the illumination map, which is then combined with the reflectance map to reconstruct a visually improved image.  
This method avoids reliance on ground-truth reflectance and instead uses paired data for training.  
![Retinex-Net Architecture](retinex_net.png)

---

#### **URetinex-Net**  
**URetinex-Net** introduces a Retinex-based **deep unfolding network** with three key modules:
1. **Initialization Module**: Sets the initial conditions for decomposition.  
2. **Unfolding Optimization Module**: Uses a deep network to iteratively optimize the decomposition into reflectance and illumination components.  
3. **Illumination Adjustment Module**: Allows user-defined adjustments to the illumination map for customized enhancements.  
By leveraging **transformers**, URetinex-Net efficiently handles long-range dependencies and produces high-quality enhancements.  
![URetinex-Net Architecture](uretinex_net.png)

---

#### **PairLIE**  
**PairLIE** adopts an unsupervised approach, training on **pairs of low-light images** to learn adaptive priors for illumination and reflectance. It uses three sub-networks:  
1. **L-Net**: Estimates the illumination map.  
2. **R-Net**: Extracts the reflectance map.  
3. **P-Net**: Removes noise and unwanted features from the original low-light image.  
During training, these networks work together to optimize illumination and reflectance, enabling robust enhancement even in challenging low-light conditions.  
![PairLIE Architecture](pairlie.png)


### **2️⃣ Dataset**
- **LOL Dataset**: Contains 500 low-light image pairs for training and testing.
- **Cube++ Dataset**: Offers 4,890 high-resolution images with varying light conditions.

### **3️⃣ Approach**
1. **Retinex Theory**: Decomposes images into illumination and reflectance components.
2. **Retinex-Net**: Enhances the illumination map using perceptual loss.
3. **Diffusion Models**: Utilize varying illumination information to learn enhancement strategies.

---

## **Results**

### **Key Achievements**
- Implementation of **Retinex-Net** for illumination enhancement.
- Integration of perceptual loss for improving visual quality.
- Comparative analysis of **PSNR** and **SSIM** values with state-of-the-art methods.

### **Sample Results**
- Enhanced images demonstrate reduced noise and improved contrast under low-light conditions.
- Qualitative and quantitative results show significant improvement over baseline methods.

---

## **Learnings**
- Gained expertise in **Retinex theory** and its application in LLIE.
- Explored advanced methods like **URetinex-Net** and **diffusion models**.
- Understood **image decomposition techniques** and **evaluation metrics** like PSNR and SSIM.

---

## **Team**
- **G.Gyaneshwar Rao**  
  *Email*: [ggyaneshwarrao1@gmail.com](mailto:ggyaneshwarrao1@gmail.com)
- **Tarun Divatagi**  
- **Rahul Hegde**  

---

## **References**
1. **Deep Retinex Decomposition for Low-Light Enhancement**: [View Paper](https://arxiv.org/abs/1808.04560)
2. **URetinex-Net: Retinex-based Deep Unfolding Network for Low-light Image Enhancement**: [View Paper](https://arxiv.org/abs/2206.03080)
3. **Diff-Retinex: Rethinking Low-Light Image Enhancement with Generative Diffusion Models**: [View Paper](https://arxiv.org/abs/2303.06705)

---
