# **Overview of Approach**

**Objective:** Develop a deep understanding of **OpenFold’s architecture**, focusing on each component, its data flow, and the mathematical transformations throughout the pipeline.

**Goals:**

- Study the architecture and its major modules.  
- Understand how biological sequences are mathematically transformed.  
- Create a visualization tool similar to LLM Visualization to interpret internal operations.

---

# **Overview: Key Representations & Data Flow**

Before analyzing individual modules, it is useful to understand the **core data representations** and how they move through the system.

### **1. Sequence / MSA Input**

- **Input:** Target amino acid sequence and a Multiple Sequence Alignment (MSA) of homologous proteins.  
- **Purpose:** Provides evolutionary context, showing which residues co-vary across related sequences.

### **2. Pair Representation**

- Encodes residue-pair relationships (e.g., co-evolution, distance, or contact probability).  
- Represented as a matrix indexed by residues *i, j*.

### **3. Single Representation**

- Per-residue features summarizing MSA and pair information for each amino acid.

### **4. Templates (Optional)**

- Incorporate structural information from homologous proteins (e.g., PDB entries) when available.

### **5. Recycling**

- Outputs (pair/single representations and coordinates) are fed back into the model for refinement.

### **6. Structure / Coordinate Output**

- Final prediction of 3D atomic coordinates (backbone and sidechain) with confidence metrics such as pLDDT.

---

# **Core Components**

## **1. Input Embedding (Sequence & MSA)**

**Purpose:** Transforms raw amino acid sequences and MSAs into dense vector representations suitable for neural network processing.

**Inputs:**

- Primary sequence (length *L*)  
- MSA of homologous sequences (shape *N × L*)  
- Optional template features

**Outputs:**

- MSA embedding: *(N, L, Cₘ)*  
- Pair representation: *(L, L, C_z)*

**Details:**

- Linear projection of one-hot amino acid encodings with positional information.  
- Shared weights across MSA sequences for parameter efficiency.  
- Pair representation initialized using outer-product mean of embeddings.  
- Outputs feed directly into the **Evoformer** module.

**Reference:** `openfold/model/embedding.py`

---

## **2. Evoformer Block**

**Purpose:** The computational backbone of OpenFold. It jointly refines MSA and pair representations through attention and geometric reasoning.

**Inputs:**

- MSA representation: *(N, L, Cₘ)*  
- Pair representation: *(L, L, C_z)*

**Outputs:**

- Updated MSA and pair representations containing contextual and geometric information.

**Key Operations:**

- **MSA Row Attention:** Captures dependencies within sequences.  
- **MSA Column Attention:** Transfers information across homologous sequences.  
- **Transition (Feedforward):** Refines MSA embeddings.  
- **Outer Product Mean:** Updates pair features.  
- **Triangle Multiplicative Update:** Models geometric consistency among residue triplets.  
- **Triangle Attention:** Learns geometric dependencies between residue pairs.  
- **Pair Transition:** Feedforward refinement of pair features.

**Notes:**

- Repeated 48–96 times in full models.  
- Uses layer normalization, residual connections, and dropout.  
- Feeds results into the **Structure Module**.

**Reference:** `openfold/model/evoformer.py`

---

## **3. Template Embedding**

**Purpose:** Integrates known 3D structural information from homologous proteins to guide folding predictions.

**Inputs:**

- Template sequences and coordinates  
- Target-template alignments

**Outputs:**

- Template-augmented pair representation enriching geometric context.

**Details:**

- Computes distances and orientations from template 3D structures.  
- Projects them into the pair-representation space.  
- Fuses with model pair features before Evoformer processing.

**Reference:** `openfold/model/embedders/template_embedder.py`

---

## **4. Recycling Mechanism**

**Purpose:** Allows iterative refinement by feeding model outputs back into the pipeline for improved accuracy.

**Inputs:**

- Previous iteration's MSA and pair representations  
- Predicted coordinates

**Process:**

- Embeds previous residue positions as geometric input.  
- Reuses prior pair representations to initialize new passes.  
- Typically runs 3–4 recycles to improve physical plausibility.

**Reference:** `openfold/model/recycling_embedder.py`

---

## **5. Structure Module**

**Purpose:** Transforms abstract residue representations into explicit 3D coordinates of atoms while maintaining geometric consistency.

**Inputs:**

- Final pair representation  
- MSA summary (usually the first MSA row)

**Outputs:**

- Predicted atomic coordinates (backbone and sidechain)

**Key Features:**

- **Invariant Point Attention (IPA):** Maintains rotational and translational invariance.  
- Predicts residue frames, backbone positions, and torsion angles.  
- Iteratively refines geometry through spatial attention.  
- Outputs are recycled for additional refinement passes.

**Reference:** `openfold/model/structure_module.py`

---

## **6. Loss Functions**

**Purpose:** Train the model to produce physically accurate and experimentally consistent protein structures.

**Main Loss:**

- **Frame-Aligned Point Error (FAPE):** Measures positional and orientational accuracy.

**Auxiliary Losses:**

- Distogram loss (residue-residue distances)  
- Angle/torsion loss  
- Violation loss (bond and angle constraints)  
- Predicted LDDT loss (local confidence estimation)

**Reference:** `openfold/loss/loss.py`

---

## **7. Inference Pipeline**

**Purpose:** Combines preprocessing, model inference, and postprocessing into a unified structure prediction pipeline.

**Inputs:**

- FASTA sequence  
- Optional templates or MSA files

**Outputs:**

- Predicted 3D structure (.pdb)  
- Confidence metrics (pLDDT, TM-score)

**Steps:**

- Generate MSAs using external tools (JackHMMER, HHblits).  
- Prepare and batch inputs for model inference.  
- Run forward passes with recycling iterations.  
- Apply structure relaxation (Amber or OpenMM).

**Reference:** `openfold/inference.py`

---

# **Summary of AlphaFold2 Insights**

### **General Overview**

- Achieves near-experimental accuracy in 3D protein structure prediction.  
- Integrates evolutionary data (MSAs) with geometric reasoning in an end-to-end model.  
- Key modules: **Evoformer**, **Invariant Point Attention**, and **Recycling**.  
- Trained on pre-2018 PDB data plus self-distilled pseudo-labels.

### **Key Mechanisms**

- **Evoformer:** Captures residue-residue relationships through attention and triangle updates.  
- **Structure Module:** Converts embeddings into 3D coordinates under geometric constraints.  
- **Recycling:** Iteratively refines outputs for improved accuracy.

### **Training**

- Uses FAPE loss plus auxiliary geometric and confidence losses.  
- Self-distillation enhances unlabeled training data.  
- Gradient checkpointing and mixed precision manage large memory demands.

### **Performance**

- Achieved best results in CASP14 with ~1 Å backbone RMSD.  
- High correlation between predicted confidence (pLDDT) and actual accuracy.  
- Performs best on single-chain, globular proteins with deep MSAs.

### **Limitations**

- Performance declines for shallow MSAs or multi-chain complexes.  
- Does not handle ligands or post-translational modifications.  
- Computationally expensive for very large proteins.  
- Predicts static conformations and misses dynamic flexibility.

### **Conceptual Insights**

- Demonstrates that deep learning can learn physical folding principles.  
- Combines biological priors with learned geometric constraints.  
- Iterative recycling parallels traditional energy minimization.  
- Inspired successors such as **OpenFold** and **ESMFold**.

---

# Mathematical Operations

## **1. Input & Embedding**

**Input:**

- Target amino acid sequence of length *L*  
- Multiple Sequence Alignment (MSA) with shape *(N_seq × L)*  
- (Optional) Template structure features

**Model initializes:**

- **MSA representation:** *M ∈ ℝ^(N_seq × L × C_m)*  
- **Pair representation:** *Z ∈ ℝ^(L × L × C_z)*

**Pair representation initialization (outer-product mean):**

This step builds the pairwise relational matrix encoding co-evolutionary signals between residues.

---

## **2. Evoformer — Joint Update of MSA and Pair Features**

Each Evoformer block updates both **MSA** and **pair representations** through attention-based and geometric reasoning.

1. **Row Attention (within MSA)**  
2. **Column Attention (across sequences):** Propagates information vertically across homologous sequences.  
3. **Outer-Product Mean (update to pair representation)**  
4. **Triangle Multiplicative Update:** Enforces geometric consistency among residue triplets (i, j, k).  
5. **Feed-Forward Transitions:** Each submodule uses layer normalization, dropout, and residual connections.

---

## **3. Structure Module**

Transforms abstract residue embeddings into **3D coordinates** of atoms.  
Each residue *i* is represented by a local frame *(R_i, t_i)*, consisting of rotation and translation components.

**Invariant Point Attention (IPA):**

This operation is invariant to rotation and translation, ensuring geometric consistency.

**Outputs:**

- Residue frames (local coordinate systems per residue)  
- Backbone atom positions  
- Side-chain torsion angles  

These are combined to yield the full atomic structure of the protein.

---

## **4. Loss Functions**

The network is trained end-to-end using multiple geometric and statistical loss functions.

**Main Loss — Frame-Aligned Point Error (FAPE):**

Ensures correct relative residue orientations regardless of absolute position.

**Auxiliary Losses:**

- **Distogram Loss:** Cross-entropy on predicted vs. true residue-residue distance bins  
- **Angle Loss:** Penalizes torsion angle prediction errors  
- **Violation Loss:** Ensures bond length and angle consistency  
- **Predicted LDDT Loss:** Encourages accurate confidence estimation  

The total loss is a weighted sum of all these terms.

---

## **5. Recycling**

After one forward pass, AlphaFold reuses its own outputs (embeddings and predicted structure) as inputs for further refinement.

**Recycling step:**

- Previous *MSA* and *pair representations* are re-embedded  
- Predicted structure coordinates are encoded geometrically and re-fed into the network  

This process typically repeats **3–4 times**, progressively improving accuracy.  
Mathematically, recycling can be viewed as iterative fixed-point refinement over the function space of structure predictions.
