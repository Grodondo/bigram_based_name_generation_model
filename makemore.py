import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


N_ELEMENTS = 27  # 26 letters + 1 for the dot


def main():
    print("-- Bigram word generator model --")

    words = open("names.txt", "r").read().splitlines()
    print(words[:10:])
    print("Number of words:", len(words))

    N = torch.zeros((N_ELEMENTS, N_ELEMENTS), dtype=torch.int32)
    chars = sorted(list(set("".join(words))))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}

    # region -- Bruteforce --
    print("-- Bruteforce method, first aprocach --")

    for w in words[:]:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    # show_bigrams(N, itos, figsize=(16, 16))

    # Normalize the matrix by rows (we add a count of n so that no element is zero)
    P = (N + 1).float()
    P /= P.sum(dim=1, keepdim=True)

    g = torch.Generator().manual_seed(2147483647)

    out = []
    for i in range(10):
        ix = 0
        while True:
            # Take the row corresponding to the current character and normalize its value
            p = P[ix]
            # Sample the next character
            ix = torch.multinomial(p, 1, replacement=True, generator=g).item()
            # print(ix, " ", p[ix].item(), " ", itos[ix])
            out.append(itos[ix])
            # If we reach the end of the word, break
            if ix == 0:
                break
            # -- print(out -> guessed words) --
    for word_guess in "".join(out).split("."):
        print(word_guess)

    log_likelihood = 0
    n = 0

    for w in words[:1]:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            logprob = torch.log(prob)
            log_likelihood += torch.log(P[ix1, ix2])
            n += 1
            print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")

    print("Log-likelihood:", log_likelihood)
    nll = -log_likelihood
    print(f"{nll=:.4f}")
    print(f"{nll / n:.4f}")
    # endregion

    # region -- TRAINING Neural Net--
    print("-- Training Neural Net, second approach --")
    xs, ys = [], []

    for w in words[:]:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)

    num_examples = len(xs)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    print(f"{num_examples=}")

    # * Initialize the weights for the nn
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn(N_ELEMENTS, N_ELEMENTS, generator=g, requires_grad=True)  # weights

    for k in range(100):
        # * Forward pass
        xenc = F.one_hot(xs, num_classes=N_ELEMENTS).float()  # Input to the nn: one-hot
        logits = xenc @ W  # Multiply the one-hot encoding by the weights

        # * Softmax the logits to get the probabilities
        counts = logits.exp()  # Exponentiate the logits to get the counts
        probs = counts / counts.sum(dim=1, keepdim=True)  # Normalize the counts

        # * Loss function
        # fmt: off
        loss = -probs[torch.arange(len(ys)), ys].log().mean() + ( # forces the weidhts towards the inteded value
            0.01 * (W**2).mean() # Controls the force at which the weights are pulled towards zero
        )  # Negative log likelihood
        # fmt: on

        # * Backward pass (compute the gradients)
        W.grad = None  # Reset the gradients
        loss.backward()  # Compute the gradients

        # * Update the weights
        W.data -= 50 * W.grad
    print(f"{k=}, {loss=:.4f}")

    # Notes -> Array W will eventually become the matrix P and loss will always try to reach the value nll / n:.4f form above

    # * ------------------------- Show words -------------------------
    g = torch.Generator().manual_seed(2147483647)

    out = []
    for i in range(10):
        ix = 0
        while True:

            xenc = F.one_hot(torch.tensor([ix]), num_classes=N_ELEMENTS).float()
            logits = xenc @ W

            # * Softmax the logits to get the probabilities
            counts = logits.exp()
            p = counts / counts.sum(dim=1, keepdim=True)

            # Sample the next character
            ix = torch.multinomial(p, 1, replacement=True, generator=g).item()
            # print(ix, " ", p[ix].item(), " ", itos[ix])
            out.append(itos[ix])
            # If we reach the end of the word, break
            if ix == 0:
                break
    # -- print(out -> guessed words) --
    for word_guess in "".join(out).split("."):
        print(word_guess)

    # endregion


def show_bigrams(N, itos, figsize=(16, 16)):
    plt.figure(figsize=figsize)
    plt.imshow(N, cmap="hot")
    for i in range(N_ELEMENTS):
        for j in range(N_ELEMENTS):
            chstr = f"{itos[i]} {itos[j]}"
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

    plt.axis("off")  # Turn off the axis for better readability
    plt.show()


# A function to show the training showcase, only needed for understanding the training process
def training_showcase(xs, ys, itos, probs):
    nlls = torch.zeros(5)
    for i in range(5):
        # i-th bigram:
        x = xs[i].item()  # input character index
        y = ys[i].item()  # label character index
        print("--------")
        print(f"bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x},{y})")
        print("input to the neural net:", x)
        print("output probabilities from the neural net:", probs[i])
        print("label (actual next character):", y)
        p = probs[i, y]
        print("probability assigned by the net to the the correct character:", p.item())
        logp = torch.log(p)
        print("log likelihood:", logp.item())
        nll = -logp
        print("negative log likelihood:", nll.item())
        nlls[i] = nll

    print("=========")
    print("average negative log likelihood, i.e. loss =", nlls.mean().item())


if __name__ == "__main__":
    main()
