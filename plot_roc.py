import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, accuracy_score

plt.rc('font', family='serif')

y_true_vanilla, y_pred_vanilla = np.loadtxt('true_vs_pred_plain.txt', unpack=True)
y_true_dc, y_pred_dc = np.loadtxt('true_vs_pred_dc.txt', unpack=True)
y_true_se, y_pred_se = np.loadtxt('true_vs_pred_se.txt', unpack=True)
y_true_all, y_pred_all = np.loadtxt('true_vs_pred_dc_se.txt', unpack=True)

y_true_triage, y_pred_triage = np.loadtxt('/Users/liangyu/Documents/astronet-triage/true_vs_pred_all.txt', unpack=True)

p_v, r_v, _ = precision_recall_curve(y_true_vanilla, y_pred_vanilla)
ap_v = average_precision_score(y_true_vanilla, y_pred_vanilla)
auc_v = roc_auc_score(y_true_vanilla, y_pred_vanilla)
acc_v = accuracy_score(y_true_vanilla, y_pred_vanilla > 0.5)
print 'vetting plain:', ap_v, auc_v, acc_v
p_dc, r_dc, _ = precision_recall_curve(y_true_dc, y_pred_dc)
ap_dc = average_precision_score(y_true_dc, y_pred_dc)
auc_dc = roc_auc_score(y_true_dc, y_pred_dc)
acc_dc = accuracy_score(y_true_dc, y_pred_dc > 0.5)
print "depth change:", ap_dc, auc_dc, acc_dc
# p_tmag, r_tmag, _ = precision_recall_curve(y_true_tmag, y_pred_tmag)
# auc_tmag = auc(p_tmag, r_tmag)
p_se, r_se, _ = precision_recall_curve(y_true_se, y_pred_se)
ap_se = average_precision_score(y_true_se, y_pred_se)
auc_se = roc_auc_score(y_true_se, y_pred_se)
acc_se = accuracy_score(y_true_se, y_pred_se > 0.5)
print "secondary eclipse:", ap_se, auc_se, acc_se
p_all, r_all, _ = precision_recall_curve(y_true_all, y_pred_all)
ap_all = average_precision_score(y_true_all, y_pred_all)
auc_all = roc_auc_score(y_true_all, y_pred_all)
acc_all = accuracy_score(y_true_all, y_pred_all > 0.5)
print "vetting all:", ap_all, auc_all, acc_all

p_triage, r_triage, _ = precision_recall_curve(y_true_triage, y_pred_triage)
ap_triage = average_precision_score(y_true_triage, y_pred_triage)
auc_triage = roc_auc_score(y_true_triage, y_pred_triage)
acc_triage = accuracy_score(y_true_triage, y_pred_triage > 0.5)
print "triage:", ap_triage, auc_triage, acc_triage

fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(111)
ax1.minorticks_on()
ax1.tick_params(top=True, right=True, which='both', direction='in')
plt.gca().xaxis.set_tick_params(labelsize=15)
plt.gca().yaxis.set_tick_params(labelsize=15)
plt.plot(r_v, p_v, label='Vetting - plain', lw=3)
plt.plot(r_dc, p_dc, label='Vetting - depth change', ls='--')
# plt.plot(r_tmag, p_tmag, label='Vetting - Tmag', ls='--')
plt.plot(r_se, p_se, label='Vetting - secondary eclipse', ls='--')
plt.plot(r_all, p_all, label='Vetting - depth change + secondary eclipse', lw=3)
plt.plot(r_triage, p_triage, label='Triage', lw=3, ls='-.')
plt.ylabel('Precision', fontsize=15)
plt.xlabel('Recall', fontsize=15)
plt.xlim(0, 1.01)
# plt.yrange(0.4,1)
plt.legend(fontsize=15, frameon=False)
plt.savefig('precision_recall.pdf', dpi=150)