import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analizar_longitud_tokens(textos, tokenizer, max_len_actual=512, prefix='', verbose=False,add_verbose = False):
    """
    Calcula estadísticas sobre la cantidad de tokens por muestra.
    Incluye los tokens del prefijo nomic en la cuenta.
    Si verbose=True, imprime cada muestra que supera max_len_actual.
    """
    print('⏳ Calculando longitudes de tokens reales (sin truncar)...')
    prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
    longitudes = [
        len(tokenizer.encode(str(t), add_special_tokens=False)) + prefix_len + 2
        for t in textos
    ]
    s_longitudes = pd.Series(longitudes)
    stats = {
        'Promedio':        s_longitudes.mean(),
        'Desv. Estándar':  s_longitudes.std(),
        'Mínimo':          s_longitudes.min(),
        'Q1 (25%)':        s_longitudes.quantile(0.25),
        'Mediana (50%)':   s_longitudes.median(),
        'Q3 (75%)':        s_longitudes.quantile(0.75),
        'Percentil 90':    s_longitudes.quantile(0.90),
        'Percentil 95':    s_longitudes.quantile(0.95),
        'Percentil 99':    s_longitudes.quantile(0.99),
        'Máximo':          s_longitudes.max(),
    }
    print('\n📊 ESTADÍSTICAS DE LONGITUD DE TOKENS (prefijo incluido)')
    print('═' * 45)
    for k, v in stats.items():
        print(f'{k:<20}: {v:>8.0f}')

    superan_max = (s_longitudes > max_len_actual).sum()
    pct_superan = (superan_max / len(s_longitudes)) * 100
    print('─' * 45)
    print(f'Muestras > {max_len_actual} tokens : {superan_max} ({pct_superan:.2f}%)')

    cont = 0
    if verbose:
        casos = [(i, longitudes[i], str(textos[i]),add_verbose[i]) for i in range(len(textos)) if longitudes[i] > max_len_actual]
        if casos:
            print(f'\n🔍 CASOS QUE SUPERAN {max_len_actual} TOKENS')
            print('═' * 60)
            for idx, tokens, texto,add_ver in casos:
                preview = texto[:80] + '...' if len(texto) > 80 else texto

                if add_ver:
                    print(f'  [{idx:>5}] {tokens:>5} tokens | verbose {add_ver}│ {preview}')
                    cont += 1
            print('═' * 60)
            print("Total verbose: ",cont)
        else:
            print('\n✅ Ninguna muestra supera el límite de tokens.')

    plt.figure(figsize=(8, 4))
    sns.histplot(s_longitudes, bins=50, kde=True, color='#3498db')
    plt.axvline(max_len_actual, color='red', linestyle='--',
                label=f'Límite actual ({max_len_actual})')
    plt.title('Distribución de Longitud de Tokens (con prefijo nomic)')
    plt.xlabel('Cantidad de Tokens')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    plt.savefig('distribucion_tokens.png', dpi=150)
    print('📈 Histograma guardado en distribucion_tokens.png\n')

    return s_longitudes, stats