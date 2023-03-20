import custom_taxi
import numpy as np


# def encode(taxi_row, taxi_col, pass_loc, dest_idx):
#     # (20) 20, 5, 4
#     i = taxi_row
#     i *= 20
#     i += taxi_col
#     i *= 5
#     i += pass_loc
#     i *= 4
#     i += dest_idx
#     return i
#
#
#
# def decode( i):
#     out = []
#     out.append(i % 4)
#     i = i // 4
#     out.append(i % 5)
#     i = i // 5
#     out.append(i % 20)
#     i = i // 20
#     out.append(i)
#     assert 0 <= i < 8000
#     return reversed(out)

def test_encode_decode():
    env = custom_taxi.CustomTaxiEnv()
    states = np.zeros(20 * 20 * 5 * 4)
    for row in range(20):
        for col in range(20):
            for pass_pos in range(5):
                for dest_pos in range(4):
                    ans = list([row, col, pass_pos, dest_pos])
                    x = env.encode(row, col, pass_pos, dest_pos)
                    decoded = list(env.decode(x))
                    states[x] = 1

                    if decoded != ans:
                        print("Actual: ", ans)
                        print("Encoded : ", x)
                        print("Decoded: ", decoded)
                    assert decoded == ans
    expected_states = np.ones(20 * 20 * 5 * 4)
    assert np.array_equal(expected_states, states)

    print("All test cases passed")


if __name__ == '__main__':
    test_encode_decode()


