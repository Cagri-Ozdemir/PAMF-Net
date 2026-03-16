# PAMF-Net
![PAMF_Net (2)](https://github.com/user-attachments/assets/29912d43-fc99-44bb-8da8-602c961e7f9b)
Pathway-Aware Multimodal Fusion Neural Network (PAMF-Net) processes pathway-specific Single Nucleotide Variant (SNV) matrices through self-attention layers to generate pathway-aware embeddings, which are then fused via an integration layer. The integrated embeddings are combined with clinicopathological features and pass through a two-layer fully connected neural network to obtain breast cancer recurrence probabilities. The attention coefficients learned by the integration layer quantify the relative contribution of each pathway to the recurrence prediction.
# How to run
Run [PAMF-Net.py](PAMF-Net.py)
You can run PAMF-Net on TCGA-BRCA and METABRIC datasets. Select the dataset to run (line 33 in [PAMF-Net.py](PAMF-Net.py)).

To plot lollipop plots illustrating log2 odds ratios of pathway mutations run :

- 
- METABRIC dataset
