## Concept-centric Personalization with Large-scale Diffusion Priors

### Abstract
>Despite large-scale diffusion models being highly capable of generating diverse open-world content, they still struggle to match the photorealism and fidelity of concept-specific generators.
In this work, we present the task of customizing large-scale diffusion priors for specific concepts as **concept-centric personalization**. Our goal is to generate high-quality concept-centric images while maintaining the versatile controllability inherent to open-world models, enabling applications in diverse tasks such as concept-centric stylization and image translation.
Distinct from existing personalization tasks that focus on specific entities, the proposed task requires large-scale training to achieve reasonable results. It brings forth unique challenges: difficult fidelity and controllability balancing and severe unconditional guidance drift in classifier-free guidance.
>
>To tackle these challenges, we identify catastrophic forgetting of guidance prediction from diffusion priors as the fundamental issue. Consequently, we develop a guidance-decoupled personalization framework specifically designed to address this task.
We propose **Generalized Classifier-free Guidance** (GCFG) as the foundational theory for our framework. This approach extends Classifier-free Guidance (CFG) to accommodate an arbitrary number of guidances, sourced from a variety of conditions and models. 
Employing GCFG enables us to separate conditional guidance into two distinct components: concept guidance for fidelity and control guidance for controllability. This division makes it feasible to train a specialized model for concept guidance, while ensuring both control and unconditional guidance remain intact.
We then present a null-text \textbf{Concept-centric Diffusion Model} as a concept-specific generator to learn concept guidance without the need for text annotations.
Furthermore, we demonstrate the seamless integration of our framework with existing techniques and underscore the vast potential of GCFG. Our extensive experiments confirm the superior efficacy of our proposed method in achieving concept-centric personalization.

### Coming Soon