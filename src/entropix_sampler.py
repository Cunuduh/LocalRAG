import torch
from torch.nn import functional as F
import numpy as np
from llama_cpp import LogitsProcessor
from typing import List, Tuple
from dataclasses import dataclass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

def calculate_varentropy_logsoftmax(logits: np.ndarray, axis: int = -1) -> Tuple[float, float]:
	"""Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
	logits_tensor = torch.from_numpy(logits).to(device)
	log_probs = F.log_softmax(logits_tensor, dim=axis)
	probs = torch.exp(log_probs)
	entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
	varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
	return entropy.item(), varentropy.item()

def _sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int, min_p: float) -> int:
	logits_tensor = torch.from_numpy(logits).to(device)
	probs = F.softmax(logits_tensor / temperature, dim=-1)

	if min_p > 0.0:
		p_max = torch.max(probs)
		probs[probs < (min_p * p_max)] = 0
		probs = probs / probs.sum()

	top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
	
	cumulative_probs = torch.cumsum(top_k_probs, dim=-1)
	probs_to_keep = cumulative_probs <= top_p
	if not probs_to_keep.any():
		probs_to_keep[-1] = True
	top_k_probs = top_k_probs[probs_to_keep]
	top_k_indices = top_k_indices[probs_to_keep]

	if top_k_probs.sum() <= 0:
		return torch.argmax(probs).item()

	try:
		sample = torch.multinomial(top_k_probs, num_samples=1)
		return top_k_indices[sample].item()
	except RuntimeError:
		return torch.argmax(probs).item()

@dataclass
class SamplerConfig:
	temp: float = 0.666
	top_p: float = 0.90
	top_k: int = 27
	min_p: float = 0.03
	low_ent_thresh: float = 0.1
	low_vent_thresh: float = 0.1
	med_ent_thresh: float = 3.0
	high_ent_thresh: float = 5.0
	high_vent_thresh: float = 5.0
	helv_attn_ent_offset: float = 1.3
	helv_attn_ent_coef: float = 0.2
	lehv_interaction_strength_offset: float = 1.2
	lehv_interaction_strength_coef: float = 0.3
	hehv_attn_ent_coef: float = 0.2
	hehv_attn_vent_offset: float = 2.0
	hehv_attn_vent_coef: float = 0.5
	n_adaptive_samples: int = 5
	ada_temp_logits: float = 0.3
	ada_temp_attn: float = 0.2
	ada_temp_agree: float = 0.2
	ada_top_p: float = 0.1
	ada_top_k_int: float = 0.3
	ada_top_k_agree: float = 0.2
	ada_min_p: float = 0.5
	ada_score_logits_ent: float = 0.1
	ada_score_attn_ent: float = 0.2
	ada_score_logits_vent: float = 0.3
	ada_score_attn_vent: float = 0.4
	ada_score_agree: float = 0.5
	ada_score_int: float = 0.6

class EntropixLogitsProcessor(LogitsProcessor):
	def __init__(self, config: SamplerConfig):
		self.config = config
		self.clarifying_question_token = 2564  # You may need to adjust this value

	def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
		logits = np.array(scores)
		attention_scores = np.random.rand(1, 1, len(input_ids), len(input_ids))  # Dummy attention scores
		
		entropy, varentropy = calculate_varentropy_logsoftmax(logits)
		
		if entropy < self.config.low_ent_thresh and varentropy < self.config.low_vent_thresh:
			sampled_token = np.argmax(logits)
		elif entropy > self.config.high_ent_thresh and varentropy < self.config.low_vent_thresh:
			if self.clarifying_question_token not in input_ids[-1:]:
				sampled_token = self.clarifying_question_token
			else:
				temp_adj = self.config.helv_attn_ent_offset + self.config.helv_attn_ent_coef * np.mean(attention_scores)
				sampled_token = _sample(logits, temperature=min(1.5, self.config.temp * temp_adj),
										top_p=self.config.top_p, top_k=self.config.top_k, min_p=self.config.min_p)
		elif entropy < self.config.high_ent_thresh and varentropy > self.config.high_vent_thresh:
			temp_adj = self.config.lehv_interaction_strength_offset + self.config.lehv_interaction_strength_coef * np.mean(np.abs(attention_scores))
			top_k_adj = max(5, int(self.config.top_k * (1 + 0.5 * (1 - np.mean(np.abs(attention_scores))))))
			sampled_token = _sample(logits, temperature=min(1.5, self.config.temp * temp_adj),
									top_p=self.config.top_p, top_k=top_k_adj, min_p=self.config.min_p)
		elif entropy > self.config.med_ent_thresh and varentropy > self.config.high_vent_thresh:
			temp_adj = self.config.hehv_attn_vent_offset + self.config.hehv_attn_vent_coef * np.var(np.mean(attention_scores, axis=(0, 1)))
			top_p_adj = max(0.5, self.config.top_p - self.config.hehv_attn_ent_coef * np.mean(attention_scores))
			sampled_token = _sample(logits, temperature=max(2.0, self.config.temp * temp_adj),
									top_p=top_p_adj, top_k=self.config.top_k, min_p=self.config.min_p)
		else:
			logits_uncertainty = entropy + varentropy
			attn_uncertainty = np.mean(attention_scores) + np.var(np.mean(attention_scores, axis=(0, 1)))
			temperature = self.config.temp * (1 + self.config.ada_temp_logits * logits_uncertainty + 
												self.config.ada_temp_attn * attn_uncertainty - 
												self.config.ada_temp_agree * np.mean(np.abs(attention_scores)))
			top_p = np.clip(self.config.top_p * (1 + self.config.ada_top_p * np.var(np.mean(attention_scores, axis=(0, 1)))), 0.1, 1.0)
			top_k = int(np.clip(
				np.round(self.config.top_k * (1 + self.config.ada_top_k_int * np.mean(np.abs(attention_scores)) - 
												self.config.ada_top_k_agree * np.mean(np.abs(attention_scores)))),
				a_min=1,
				a_max=100
			))
			min_p = np.clip(self.config.min_p * (1 - self.config.ada_min_p * logits_uncertainty), 0.01, 0.5)
			
			samples = [_sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p) 
					 for _ in range(self.config.n_adaptive_samples)]
			
			def score_sample(sample):
				log_prob = np.sum(F.log_softmax(torch.from_numpy(logits), dim=-1).numpy() * 
									F.one_hot(torch.tensor(sample), num_classes=logits.shape[-1]).numpy())
				confidence_score = (
					(1 - entropy) * self.config.ada_score_logits_ent +
					(1 - np.mean(attention_scores)) * self.config.ada_score_attn_ent +
					(1 - varentropy) * self.config.ada_score_logits_vent +
					(1 - np.var(np.mean(attention_scores, axis=(0, 1)))) * self.config.ada_score_attn_vent +
					np.mean(np.abs(attention_scores)) * self.config.ada_score_agree +
					np.mean(np.abs(attention_scores)) * self.config.ada_score_int
				)
				return log_prob + confidence_score
			
			sample_scores = [score_sample(sample) for sample in samples]
			sampled_token = samples[np.argmax(sample_scores)]

		new_scores = [-float('inf')] * len(scores)
		new_scores[sampled_token] = 0
		return new_scores
