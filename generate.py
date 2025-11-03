def generate(model, sp, start_text="こんにちは", max_new_tokens=50):
    model.eval()
    ids = torch.tensor(sp.encode(start_text), dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(ids)
        next_id = torch.argmax(logits[0, -1])
        ids = torch.cat([ids, next_id.unsqueeze(0).unsqueeze(0)], dim=1)
    return sp.decode(ids[0].tolist())

print(generate(model, sp, "AIは"))

