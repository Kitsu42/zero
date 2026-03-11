def test_prediction(nn, scale, sequence):

    # pegar últimos valores da sequência
    a = sequence[-2]
    b = sequence[-1]

    test_input = [[a, b]]

    # normalizar
    test_input = np.array(test_input, dtype=float) / scale

    # prever
    prediction = nn.predict(test_input)

    # voltar escala original
    prediction = prediction * scale

    print("\nÚltimos valores:", a, b)
    print("Próximo valor previsto:", prediction[0][0])