def plot_totals(dates, actual, prediction, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.plot(dates, actual.sum(axis=0), label="Observed units", color="#253b4a")
    axis.plot(dates, prediction.sum(axis=0), label="Global LightGBM", color="#258d78")
    axis.set(xlabel="Holdout day", ylabel="Total daily units", title="Retail demand: 28-day holdout")
    axis.legend(frameon=False); fig.tight_layout(); fig.savefig(output, dpi=170); plt.close(fig)
