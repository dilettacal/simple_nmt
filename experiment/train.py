import torch


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg
        start_src = src.shape
        start_trg = trg.shape

        optimizer.zero_grad()

        output = model(src, trg)
        raw_o = output

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]
        print("output 0:", output[0])  # a tensor full of 0
        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

        if i == 0:
            print("Start src and trg shape:")
            print(start_src, start_trg)
            print("Source and target shape:")
            print((src.shape, trg.shape))
            print("Raw output from model")
            print(raw_o.shape)
            print("Reshaped output")
            print(output.shape)

    return epoch_loss / len(iterator)