from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import AsyncIterable
from pydantic import BaseModel

from entropix.main import (
    LLAMA_1B_PARAMS,
    load_weights,
    Tokenizer,
    build_attn_mask,
    precompute_freqs_cis,
    KVCache,
    xfmr,
    sample,
)

import jax
import jax.numpy as jnp

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model weights and tokenizer
model_params = LLAMA_1B_PARAMS
xfmr_weights = load_weights()
tokenizer = Tokenizer('entropix/tokenizer.model')

class PromptRequest(BaseModel):
    prompt: str

async def generate_stream(prompt: str) -> AsyncIterable[str]:
    raw_tokens = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
    tokens = jnp.array([raw_tokens], jnp.int32)
    bsz, seqlen = tokens.shape
    cur_pos = 0
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    
    logits, kvcache = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    gen_tokens = next_token
    
    yield f"{tokenizer.decode([next_token.item()])}\n"
    
    cur_pos = seqlen
    stop = jnp.array([128001, 128008, 128009])
    
    while cur_pos < 2048:
        cur_pos += 1
        logits, kvcache = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
        next_token, ent, vent = sample(gen_tokens, logits)
        gen_tokens = jnp.concatenate((gen_tokens, next_token))
        
        token_str = tokenizer.decode(next_token.tolist()[0])
        yield f"{token_str}\t{ent:.2f}\t{vent:.2f}\n"
        
        if jnp.isin(next_token, stop).any():
            break

    yield

@app.post("/generate")
async def generate(request: PromptRequest):
    return StreamingResponse(generate_stream(request.prompt), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
