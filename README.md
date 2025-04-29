# EnerGIS-AI-Classification-Retrieval-and-Contrastive-Learning
A lightweight multimodal AI pipeline for classifying and describing power plants using satellite imagery and structured metadata. This research integrates supervised contrastive learning, image-text retrieval (FAISS), and retrieval-augmented generation (RAG) to produce interpretable and scalable outputs — optimized for energy infrastructure domains.

## 🔍 Project Summary

This project builds on prior work (AAAI 2025) that used GIS-guided models with CNNs and Vision Transformers. The new system:
- Learns semantically aligned embeddings via contrastive learning (ViT + RoBERTa)
- Retrieves similar plants using FAISS (image-to-text and text-to-image)
- Generates natural language summaries using LLMs (Groq/DeepSeek)

## 🚀 Features

- Contrastive Learning with ViT-B/16 and RoBERTa
- UMAP visualizations for interpretability
- RAG pipeline for fuel-type, capacity, and region descriptions
- Comparison against CLIP and GIS-masked CNN+ViT (from AAAI)

## 🧠 Model Comparison

| Model                    | Recall@1 | Recall@5 | mAP@5  | Notes                                      |
|-------------------------|----------|----------|--------|--------------------------------------------|
| CNN+ViT+GIS Masks (AAAI) | ~20%     | ~50%     | –      | Overfitting on spatial masks               |
| CLIP (Zero-Shot)         | <20%     | ~45%     | –      | Poor domain alignment                      |
| **Contrastive + RAG**    | **28.57%** | **68.64%** | **0.798** | Best performance and interpretability     |

## 📊 Sample Outputs

| Satellite Image | Retrieved Captions | RAG Summary |
|----------------|--------------------|-------------|
| `naip_1012_CA_WND.tif` | Similar wind farms in NE & KS | "A wind-powered facility in Nebraska producing 283,000 MWh/year" |


## 📦 Dependencies

- `torch`, `transformers`, `umap-learn`
- `faiss`, `scikit-learn`, `matplotlib`
- `deepseek`, `groq`, or similar LLM access

Use the `requirements.txt` or `environment.yml` file (coming soon) to install.

## ✨ Acknowledgments

Thanks to the Clean Energy and Sustainability Analytics Center and the Department of Computer Science at Montclair State University. Special appreciation to Dr. Aparna Varde and Dr. Hao Liu for supervision and support.

## 📖 Citation

If you use this code or framework in your research, please cite:

```bibtex
@inproceedings{austin2025kgml,
  author    = {Blessing Austin-Gabriel and Aparna S. Varde},
  title     = {A Multimodal AI Framework for Power Plant Understanding via Contrastive Retrieval and Natural Language Generation},
  booktitle = {MSU Data Science Thesis, 2025},
  year      = {2025}
}

