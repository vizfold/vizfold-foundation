import React from "react";

export default function NotesPanel() {
  return (
    <div className="notes-wrapper">
      <h2 className="chapter-heading">Notes</h2>

      <div className="notes-layout">
        {/* Left: small scratchpad */}
        <div className="notes-left">
          <p className="notes-hint">Quick scratch notes or TODOs.</p>
          <textarea
            className="notes-textarea"
            placeholder="Write notes here..."
          />
        </div>

        {/* Right: main reference notes */}
        <div className="notes-right">
          <h3 className="notes-section-title">Overview of Approach</h3>

          <p>
            <strong>Objective:</strong> Develop a deep understanding of
            OpenFold&apos;s architecture, focusing on each component, its data
            flow, and the mathematical transformations throughout the pipeline.
          </p>

          <p>
            <strong>Goals:</strong>
          </p>
          <ul>
            <li>Study the architecture and its major modules.</li>
            <li>
              Understand how biological sequences are mathematically
              transformed.
            </li>
            <li>
              Create a visualization tool similar to LLM Visualization to
              interpret internal operations.
            </li>
          </ul>

          <h3 className="notes-section-title">
            Overview: Key Representations &amp; Data Flow
          </h3>
          <ul>
            <li>
              <strong>Sequence / MSA Input:</strong> Target amino acid sequence
              and an MSA of homologous proteins; provides evolutionary context
              and co-variation.
            </li>
            <li>
              <strong>Pair Representation:</strong> Matrix indexed by residues
              (i, j), encoding residue–pair relationships such as co-evolution,
              distance, or contact probability.
            </li>
            <li>
              <strong>Single Representation:</strong> Per-residue features
              summarizing MSA and pair information for each amino acid.
            </li>
            <li>
              <strong>Templates (optional):</strong> Use structural information
              from homologous proteins when available.
            </li>
            <li>
              <strong>Recycling:</strong> Feeds pair/single representations and
              predicted coordinates back into the model for refinement.
            </li>
            <li>
              <strong>Structure / Coordinate Output:</strong> Final prediction
              of 3D atomic coordinates (backbone and sidechain) plus confidence
              metrics such as pLDDT.
            </li>
          </ul>

          <h3 className="notes-section-title">Core Components</h3>

          <p>
            <strong>1. Input Embedding (Sequence &amp; MSA)</strong>
          </p>
          <ul>
            <li>
              <strong>Purpose:</strong> Transform raw amino acid sequences and
              MSAs into dense vector representations.
            </li>
            <li>
              <strong>Inputs:</strong> primary sequence (length L), MSA of
              homologous sequences (N × L), optional template features.
            </li>
            <li>
              <strong>Outputs:</strong> MSA embedding (N, L, C_m) and pair
              representation (L, L, C_z).
            </li>
            <li>
              Uses linear projection + positional encoding; weights are shared
              across MSA sequences; pair representation is initialized using an
              outer-product mean of MSA embeddings. Outputs feed into the
              Evoformer.
            </li>
          </ul>

          <p>
            <strong>2. Evoformer Block</strong>
          </p>
          <ul>
            <li>
              <strong>Purpose:</strong> Backbone of OpenFold, jointly refining
              MSA and pair representations.
            </li>
            <li>
              <strong>Inputs:</strong> MSA representation (N, L, C_m), pair
              representation (L, L, C_z).
            </li>
            <li>
              <strong>Key operations:</strong> MSA row attention, MSA column
              attention, transition (feedforward), outer product mean, triangle
              multiplicative update, triangle attention, pair transition.
            </li>
            <li>
              Repeated many times (e.g., 48–96 blocks); uses layer norm,
              residual connections, dropout; outputs are passed to the Structure
              Module.
            </li>
          </ul>

          <p>
            <strong>3. Template Embedding</strong>
          </p>
          <ul>
            <li>
              Integrates 3D information from template structures to guide
              folding predictions.
            </li>
            <li>Inputs: template sequences, coordinates, alignments.</li>
            <li>
              Outputs: template-augmented pair representation with geometric
              context, fused with the model&apos;s pair features.
            </li>
          </ul>

          <p>
            <strong>4. Recycling Mechanism</strong>
          </p>
          <ul>
            <li>
              Allows iterative refinement by feeding previous iteration outputs
              back into the network.
            </li>
            <li>
              Embeds previous residue positions and reuses previous pair
              representations.
            </li>
            <li>Typically runs 3–4 recycles to improve physical plausibility.</li>
          </ul>

          <p>
            <strong>5. Structure Module</strong>
          </p>
          <ul>
            <li>
              <strong>Purpose:</strong> Converts high-level representations into
              explicit 3D coordinates.
            </li>
            <li>
              Uses Invariant Point Attention (IPA) to maintain rotational and
              translational invariance.
            </li>
            <li>
              Predicts residue frames, backbone positions, and sidechain torsion
              angles; iteratively refines geometry.
            </li>
          </ul>

          <p>
            <strong>6. Loss Functions</strong>
          </p>
          <ul>
            <li>
              <strong>Main loss:</strong> Frame-Aligned Point Error (FAPE) for
              positional/orientational accuracy in residue frames.
            </li>
            <li>
              <strong>Auxiliary losses:</strong> distogram loss (distance
              bins), torsion/angle loss, violation loss (bond/angle
              constraints), predicted LDDT loss (confidence).
            </li>
          </ul>

          <p>
            <strong>7. Inference Pipeline</strong>
          </p>
          <ul>
            <li>
              <strong>Inputs:</strong> FASTA sequence, optional templates/MSAs.
            </li>
            <li>
              <strong>Outputs:</strong> predicted 3D structure (.pdb) plus
              confidence metrics (pLDDT, TM-score).
            </li>
            <li>
              Steps: generate MSAs (e.g., JackHMMER, HHblits), prepare inputs,
              run forward passes with recycling, optionally relax structure with
              Amber/OpenMM.
            </li>
          </ul>

          <h3 className="notes-section-title">Summary of AlphaFold2 Insights</h3>

          <p>
            <strong>General Overview</strong>
          </p>
          <ul>
            <li>Achieves near-experimental accuracy in many cases.</li>
            <li>
              Integrates evolutionary data (MSAs) with geometric reasoning in an
              end-to-end model.
            </li>
            <li>
              Key modules: Evoformer, Invariant Point Attention, and Recycling.
            </li>
          </ul>

          <p>
            <strong>Key Mechanisms</strong>
          </p>
          <ul>
            <li>
              Evoformer captures residue–residue relationships via attention and
              triangle updates.
            </li>
            <li>
              Structure module converts embeddings into 3D coordinates under
              geometric constraints.
            </li>
            <li>
              Recycling iteratively refines outputs for improved accuracy.
            </li>
          </ul>

          <p>
            <strong>Training</strong>
          </p>
          <ul>
            <li>
              Uses FAPE as the main loss plus auxiliary geometric and confidence
              losses.
            </li>
            <li>
              Self-distillation generates pseudo-labels to expand training data.
            </li>
            <li>
              Gradient checkpointing and mixed precision manage memory usage.
            </li>
          </ul>

          <p>
            <strong>Performance</strong>
          </p>
          <ul>
            <li>
              CASP14: state-of-the-art with ~1 Å backbone RMSD for many targets.
            </li>
            <li>
              High correlation between predicted confidence (pLDDT) and actual
              accuracy.
            </li>
            <li>
              Performs best on single-chain, globular proteins with deep MSAs.
            </li>
          </ul>

          <p>
            <strong>Limitations</strong>
          </p>
          <ul>
            <li>Performance drops for shallow MSAs or multi-chain complexes.</li>
            <li>Does not directly handle ligands or post-translational modifications.</li>
            <li>
              Predicts mostly static conformations; limited modeling of
              flexibility and dynamics.
            </li>
          </ul>

          <p>
            <strong>Conceptual Insights</strong>
          </p>
          <ul>
            <li>
              Demonstrates that deep learning can learn physical folding
              principles from data.
            </li>
            <li>
              Combines biological priors (MSAs, templates) with learned
              geometric constraints.
            </li>
            <li>
              Iterative recycling parallels traditional energy minimization.
            </li>
            <li>
              Inspired successors such as OpenFold and ESMFold.
            </li>
          </ul>

          <h3 className="notes-section-title">Mathematical Operations</h3>

          <p>
            <strong>1. Input &amp; Embedding</strong>
          </p>
          <ul>
            <li>
              Inputs: sequence of length L, MSA (N × L), optional templates.
            </li>
            <li>
              Initializes MSA representation M ∈ R^(N × L × C_m) and pair
              representation Z ∈ R^(L × L × C_z).
            </li>
            <li>
              Pair initialization via outer-product mean encodes co-evolutionary
              signals between residues.
            </li>
          </ul>

          <p>
            <strong>2. Evoformer Updates</strong>
          </p>
          <ul>
            <li>Row attention operates along sequence positions.</li>
            <li>Column attention propagates information across sequences.</li>
            <li>
              Outer-product mean and triangle multiplicative updates refine Z
              and enforce geometric consistency among residue triplets.
            </li>
          </ul>

          <p>
            <strong>3. Structure Module</strong>
          </p>
          <ul>
            <li>
              Predicts a local frame (rotation + translation) for each residue.
            </li>
            <li>
              Invariant Point Attention ensures operations are invariant to
              global rotation and translation.
            </li>
            <li>
              Outputs backbone atom positions and side-chain torsion angles.
            </li>
          </ul>

          <p>
            <strong>4. Loss Functions</strong>
          </p>
          <ul>
            <li>
              FAPE measures frame-aligned distance between predicted and true
              atom positions.
            </li>
            <li>
              Distogram, angle, violation, and pLDDT losses encourage physically
              realistic geometry and good confidence calibration.
            </li>
          </ul>

          <p>
            <strong>5. Recycling</strong>
          </p>
          <ul>
            <li>
              After each forward pass, updated embeddings and coordinates are
              fed back into the network as inputs.
            </li>
            <li>
              Can be viewed mathematically as an iterative fixed-point
              refinement over the space of structures.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
