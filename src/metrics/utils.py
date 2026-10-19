# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    d = [[0] * (len(predicted_text) + 1) for _ in range(len(target_text) + 1)]
    for i in range(len(target_text) + 1):
        d[i][0] = i
    for j in range(len(predicted_text) + 1):
        d[0][j] = j

    for i, t in enumerate(target_text, start=1):
        for j, p in enumerate(predicted_text, start=1):
            d[i][j] = min(
                d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + int(p != t)
            )

    return d[-1][-1] / (len(target_text) if target_text else 1)


def calc_wer(target_text, predicted_text) -> float:
    return calc_cer(target_text.split(), predicted_text.split())
