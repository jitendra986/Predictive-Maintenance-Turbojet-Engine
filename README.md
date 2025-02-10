# Predictive-Maintenance-Turbojet-Engine
Analyzed the C-MAPSS data provided by NASA and applied predictive maintenance algorithms. vanilla transformer architecture has been used to process the multivariate time series data from the C-MAPSS aircraft engine simulator to output the estimated remaining life of a component and then apply a heuristic maintenance strategy using the transformer's test data output as input. Predictive maintenance algorithms have been taken from the paper of Antonios Kamariotis's paper "A metric for assessing and optimizing data-driven prognostic algorithms for predictive maintenance." The predictive maintenance algorithms implement the following. 

# PDM Policy1

1. `Policy Overview:` We first consider the simple dynamic PdM decision setting, in which one determines at each time step
    tk whether a component should be preventively replaced or not. The assumption here is that the new
    component is readily available when a preventive replacement is decided or a corrective replacement is
    imposed. <br>
2. `Decision Making Process:`
> At each time step tk = k * ΔT, the policy decides whether to take action or not.<br>
> The action arep,k can be either:
>> a) DN (Do Nothing) <br>
>> b) PR (Preventive Replacement)<br>

3. `Decision Rule:`
> If Pr(RULpred,k ≤ ΔT) < pthres, then Do Nothing (DN) <br>
> Otherwise, perform Preventive Replacement (PR) <br>
where:
> RULpred,k is the predicted Remaining Useful Life at time tk <br>
> ΔT is the time step <br>
> pthres is a variable heuristic threshold <br>
4. `Threshold Determination:`
> Initially, pthres is set to cp/cc <br>
> cp is the cost of preventive replacement <br>
> cc is the cost of component failure <br>
5. `Cost Considerations:`
> PR action costs cp <br>
> DN action risks a potential cost of Pr(RULpred,k ≤ ΔT) * cc <br>
6. `Rationale:`
> PR is performed only when its cost is less than the predicted risk of failure in the next time step. <br>
7. `Input Requirements:`
>  `current_cycle:` variable represents the current time step or cycle within the sequence of data for a specific engine or component. It is used to iterate through the sequence of cycles for each unique engine ID in the dataset. <br>
>  The policy needs Pr(RULpred,k ≤ ΔT) from the prognostic algorithm. <br>
8. `Outcome:`
> The policy informs replacement decisions for each component. <br>
> It determines C_rep(i) (replacement cost) and Tlc(i) (lifecycle time) for each component i. <br>

>> 1. `t_LC_array(Lifecycle Time) :`
>>> * This array represents `Tlc(i) = min[T_R(i), T_F(i)]` for each component. <br>
>>> * In the code, it's set in two cases:
>>> a) When a preventive replacement is decided: `t_LC_array[counter] = params['seq_length'] + current_cycle`
>>> b) When no preventive replacement occurred (implying failure): `t_LC_array[counter] = pdm_df[pdm_df['id'] == id]['cycle'].iloc[-1]`
>>> * This aligns with the definition of `Tlc(i)` being the minimum of preventive replacement time or failure time.
>> 2. `costs_rep_array (Replacement Cost):`
>>> * This array represents C_rep(i) for each component.
>>> * In the code, it's set as follows:
>>> a) For preventive replacement: `costs_rep_array[counter] = C_p` <br>
>>> b) For corrective replacement (when no preventive replacement occurred): `costs_rep_array[counter] = C_c` <br>
>>> This directly implements the condition as mentioned in equation 8: `C_rep(i) = (cp, if T_R(i) < T_F(i), cc, else.`


9. `t_order_array:`
> * Meaning: This array stores the cycle times at which components are ordered. <br>
> * Significance: It helps track when preventive maintenance actions are initiated, allowing for analysis of the timing of maintenance decisions. <br>

10. `t_LC_array:`
> Meaning: This array likely stores the lifecycle times for each component. <br>
> Significance: It represents either the time of preventive replacement or the time of failure for each component, which is crucial for evaluating the effectiveness of the maintenance policy. <br>

11. `costs_rep_array:`
> * Meaning: This array stores the replacement costs for each component. <br>
> * Significance: It captures the financial impact of replacements, whether they are preventive (C_p) or corrective (C_c). This is essential for assessing the cost-effectiveness of the maintenance strategy. <br>

12. `costs_delay_array:`
> * Meaning: This array stores the costs associated with delays in component replacement. <br>
> * Significance: It represents the financial penalties incurred when a component fails before a replacement arrives, helping to quantify the impact of maintenance timing on overall costs. <br>

13. `costs_stock_array:`
> Meaning: This array stores the costs related to holding replacement components in stock. <br>
> Significance: It captures the inventory holding costs when components are ordered too early, balancing the trade-off between early ordering to prevent failures and the costs of storing components. <bR>

# PDM policy 2
1. `Objective:` PDM policy 2 aims to find the optimal time for preventive replacement (T_{R,k}*) by minimizing the long-run expected maintenance cost per unit time. <br>
2. `Decision Rule:` At each time step t_k, the policy decides:
> * Perform Preventive Replacement (PR) if t_k + ΔT ≥ T_{R,k}*
> * Do Nothing (DN) otherwise
3. `Optimization Problem:` The policy solves an optimization problem at each time step to find T_{R,k}* by minimizing the objective function f(T_{R,k}) given in Equation 12. <br>
4. `Components of the Objective Function (Equation 12):`
> $f\left(T_{\mathrm{R}, k}\right)=\frac{\mathrm{E}\left[C_{\mathrm{rep}}\left(T_{\mathrm{R}, k}\right)\right]}{\mathrm{E}\left[T_{\mathrm{lc}}\left(T_{\mathrm{R}, k}\right)\right]}=\frac{P_{\mathrm{PR}} \cdot c_{\mathrm{p}}+\left(1-P_{\mathrm{PR}}\right) \cdot c_{\mathrm{c}}}{P_{\mathrm{PR}} \cdot\left(T_{\mathrm{R}, k}\right)+\int_t^{T_{\mathrm{R}, k}} t f_{R U L_{\mathrm{Pred}, k}}\left(t-t_k\right) \mathrm{d} t}$
> * E[C_{rep}(T_{R,k})]: Expected cost of replacement
> * E[T_{lc}(T_{R,k})]: Expected lifecycle time
> * P_{PR}: Probability of preventive replacement (defined in Equation 13)
> * c_p: Cost of preventive replacement
> * c_c: Cost of corrective replacement (failure)
> * f_{RUL_{pred,k}}(t): Full distribution of the RUL prediction at time t_k
> * The find_optimal_replacement_time function implements the objective function from PDM policy 2. It finds the optimal replacement time T_R_k by minimizing the expected cost per unit time. <br>

5. Interpretation of Equation 12:
> * Numerator: Represents the expected cost, considering both preventive and corrective replacements.
> * Denominator: Represents the expected lifecycle time.
> * By minimizing this ratio, the policy aims to find the optimal balance between cost and component lifetime.


6. Probability of Preventive Replacement(Equation 13): 
> P_{PR} represents the probability that the component will be preventively replaced at T_{R,k}.
> It's calculated by integrating the RUL prediction distribution from T_{R,k} to infinity.

> $$
P_{\mathrm{PR}}=\int_{T_{\mathrm{R}, k}}^{\infty} f_{R U L_{\mathrm{Pred}, k}}\left(t-t_k\right) \mathrm{d} t
$$       

7. Full RUL Distribution: Unlike PDM policy 1, this policy uses the full distribution of the Remaining Useful Life (RUL) prediction, allowing for more nuanced decision-making.<br>

8. The function `find_optimal_replacement_time` and its Parameters: function is designed to determine the optimal time for preventive replacement in a Predictive Maintenance (PdM) system. It uses a lognormal distribution fitted to the predicted probabilities of component failure to calculate the expected cost per unit time and find the optimal replacement time.<br>
 , representing the probability of needing a preventive replacement after T_R_K. <br>
> `probabilities:` The predicted probabilities of class1 and class2 from the model. <br>
> C_p, C_c, current_cycle, seq_length <br>
> `Fitting the Lognormal Distribution:` `mu, sigma = fit_lognormal_cdf(probabilities):` This fits a lognormal distribution to the given probabilities, returning the parameters mu and sigma..
> * `Probability of Preventive Replacement (P_PR):` This is the probability that the component survives until T_R_k. <br>
> * `Expected Replacement Cost (E_C_rep) :` It's a weighted sum of preventive (C_p) and corrective (C_c) replacement costs.<br> 
> * `Expected Lifecycle Time (E_T_lc) :` The first term (P_PR * T_R_k) is the expected time if preventive replacement occurs
and The sum calculates the expected time if failure occurs before T_R_k.<br>

9. `Optimization:`
> * Uses scipy's `minimize_scalar` function to find the T_R_k that minimizes the objective function.<br>
> * The search is bounded between the current cycle and the end of the sequence.<br>

# PDM Policy 3 
1. `Objective: `PDM Policy 3 represents a significant advancement in predictive maintenance by utilizing the full distribution of RUL data to inform maintenance decisions. The objective function captures both the expected costs of preventive and corrective actions and the additional costs associated with early replacements, providing a comprehensive framework for optimizing maintenance strategies. <br>
2. `Explanation of Equation(14): ` The equation for the objective function in PDM Policy 3 is given as:
> $f\left(T_{\mathrm{R}, k}\right)=P_{\mathrm{PR}} \cdot C_{\mathrm{p}}+\left(1-P_{\mathrm{PR}}\right) \cdot C_{\mathrm{c}}+\int_{T_{\mathrm{R}, k}}^{\infty}\left(t-T_{\mathrm{R}, k}\right) \cdot \frac{\mathrm{E}_{\bar{T}_{\mathrm{F}}}\left[C_{\mathrm{rep}}\right]}{\mathrm{E}_{\bar{T}_{\mathrm{F}}}\left[T_{l \mathrm{c}}\right]} f_{R U L_{\mathrm{pred}, k}}\left(t-t_k\right) \mathrm{d} t$

3. `Components of the equation: `

>  1. `Expected Replacement Cost: `
>> * The first two terms, $P_{\mathrm{PR}} \cdot C_{\mathrm{p}}$ and $\left(1-P_{\mathrm{PR}}\right) \cdot C_{\mathrm{c}}$ represent the expected costs associated with preventive and corrective replacements, respectively. <br>
>> * $P_{\mathrm{PR}}$ is the probability that the components will survive until the replacement time $T_{\mathrm{R}, k}$ <br>
>> * $ C_{\mathrm{p}}$ is the cost of preventive replacement, and $ C_{\mathrm{c}}$ is the cost of corrective replacement. <br>

> 2. `Integral Term: `
>> * The integral term quantifies the additional expected maintenance cost associated with an "early" replacement at $T_{\mathrm{R}, k}$ <br>
>> * The expression $ \left(t-T_{\mathrm{R}, k}\right) $ represents the time lost due to an early replacement.
>> * The term $ \frac{\mathrm{E}_{\bar{T}_{\mathrm{F}}}\left[C_{\mathrm{rep}}\right]}{\mathrm{E}_{\bar{T}_{\mathrm{F}}}\left[T_{l \mathrm{c}}\right]} $  is the long-run expected maintenance cost per unit time concerning the distribution of the population of components. It provides a scaling factor for the additional cost incurred by replacing the component early. <br> 
>> * $ f_{R U L_{\mathrm{pred}, k}}\left(t-t_k\right) $is the predicted probability density function of the RUL, representing the likelihood of failure occurring at time t after the current cycle $t_k$



