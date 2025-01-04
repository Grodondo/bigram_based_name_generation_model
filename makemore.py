import torch
import matplotlib.pyplot as plt

N_ELEMENTS = 27  # 26 letters + 1 for the dot


def main():
    words = open("names.txt", "r").read().splitlines()
    print(words[:10:])
    print("Number of words:", len(words))

    N = torch.zeros((N_ELEMENTS, N_ELEMENTS), dtype=torch.int32)
    chars = sorted(list(set("".join(words))))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}

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

    # region -- LOSS FUNCTION --
    log_likelihood = 0
    n = 0

    for w in ["anderjq"]:
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

    # -- print(out -> guessed words) --
    for word_guess in "".join(out).split("."):
        print(word_guess)


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


if __name__ == "__main__":
    main()
