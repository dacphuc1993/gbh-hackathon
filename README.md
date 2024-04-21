# Green Battery Hackathon 2024

## Innovation Track

This is submission for GBH 2024 innovation track.

### **Data preprocessing**
-  Removed rows with missing values (299k -> 72k) to ensure dataset quality and completeness.
-  Filtered data to April and May (72k -> 17k) for relevance to the hackathon's focus period.
-  Replaced missing values in validation/live data with 0 to maintain consistency and usability of the dataset.

### **Approach**
- Utilized Soft Actor-Critic (SAC) algorithm, known for its effectiveness in reinforcement learning (RL) tasks.
- Used simple MLP network architecture with lightweight models (10MB) for efficiency, facilitating deployment and scalability.
- Achieved a score of 62, outperforming the baseline, demonstrating the algorithm's efficacy in energy trading.

### **Self-evaluation**
- **Social Good**: We believe that our optimized battery control strategy has the potential to significantly improve energy efficiency, which could lead to positive outcomes such as reduced energy prices for Australian households and lowered emissions. This aligns with broader societal goals of sustainability and cost-effectiveness in energy consumption.
- **Commercial Viability**: Considering the extensive training process involved, our solution is most viable at scale, particularly for deployment in consumer or industrial-grade battery systems. This scalability is essential for ensuring the solution's long-term sustainability and adoption in real-world energy trading scenarios.
- **Execution**: Our implementation leveraged `stable-baselines3`, a popular repository in the reinforcement learning community, to train off-the-shelf RL algorithms. While the algorithm is there to use, we have to adapt the environment to meet its input requirement. 
- **Innovation**: By using deep RL, we become one of a few teams that use Machine Learning (ML) algorithm in the final submission. One of the inherent RL challanges is sample inefficiency and slow to train effective policies. Our adaptation to achieve comparable scores within a shorter timeframe demonstrates innovation in applying advanced ML techniques to energy trading challenges. This approach opens new avenues for efficient and effective energy trading strategies.
- There is room for improvement, we have not finished training before the deadline. We only use the checkpoint of 115 episodes while the RL policies may take order of magnitude more number of episodes to train. We believe that continous training can lead to a better model.