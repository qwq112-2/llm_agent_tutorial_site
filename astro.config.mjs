// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export default defineConfig({
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [[rehypeKatex, { strict: false, output: 'html' }]],
	},
	integrations: [
		starlight({
			title: 'LLM / Agent 算法工程师手册',
			description:
				'从 0 到 LLM / Agent 算法工程师的中文教程：16 模块 / 87 节，理论 + 实现 + 工程三位一体。',
			defaultLocale: 'root',
			locales: {
				root: { label: '简体中文', lang: 'zh-CN' },
			},
			head: [
				{
					tag: 'link',
					attrs: {
						rel: 'stylesheet',
						href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
					},
				},
			],
			customCss: ['./src/styles/custom.css'],
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/qwq112-2/llm_agent_tutorial_site',
				},
			],
			pagination: true,
			lastUpdated: true,
			expressiveCode: {
				themes: ['github-dark', 'github-light'],
			},
			sidebar: [
				{
					label: '关于',
					items: [
						{ link: '/roadmap/', label: '🗺️ 学习地图（ROADMAP）' },
						{ link: '/paper-list/', label: '📚 论文清单（50+ 必读）' },
						{ link: '/interview-index/', label: '🎯 面试题索引（261 题）' },
					],
				},
				{
					label: '模块 0：引言',
					items: [{ autogenerate: { directory: '00-intro' } }],
				},
				{
					label: '模块 1：深度学习基础',
					items: [{ autogenerate: { directory: '01-dl-basics' } }],
				},
				{
					label: '模块 2：NLP 任务全景',
					items: [{ autogenerate: { directory: '02-nlp-landscape' } }],
				},
				{
					label: '模块 3：Tokenization',
					items: [{ autogenerate: { directory: '03-tokenization' } }],
				},
				{
					label: '模块 4：Transformer 从零实现 🔥',
					items: [{ autogenerate: { directory: '04-transformer-from-scratch' } }],
				},
				{
					label: '模块 5：现代 LLM 架构',
					items: [{ autogenerate: { directory: '05-modern-llm-architectures' } }],
				},
				{
					label: '模块 6：预训练',
					items: [{ autogenerate: { directory: '06-pretraining' } }],
				},
				{
					label: '模块 7：训练基础设施 🔥',
					items: [{ autogenerate: { directory: '07-training-infra' } }],
				},
				{
					label: '模块 8：后训练 SFT + PEFT',
					items: [{ autogenerate: { directory: '08-post-training-sft-peft' } }],
				},
				{
					label: '模块 9：后训练 RLHF 🔥',
					items: [{ autogenerate: { directory: '09-post-training-rlhf' } }],
				},
				{
					label: '模块 10：推理与测试时扩展',
					items: [{ autogenerate: { directory: '10-reasoning-test-time-scaling' } }],
				},
				{
					label: '模块 11：推理引擎',
					items: [{ autogenerate: { directory: '11-inference-engines' } }],
				},
				{
					label: '模块 12：评测与 LLMOps',
					items: [{ autogenerate: { directory: '12-evaluation-llmops' } }],
				},
				{
					label: '模块 13：Prompt / RAG / 工具',
					items: [{ autogenerate: { directory: '13-prompting-rag-tools' } }],
				},
				{
					label: '模块 14：Agent 系统 🔥',
					items: [{ autogenerate: { directory: '14-agent-systems' } }],
				},
				{
					label: '模块 15：Agent 强化学习 🔥',
					items: [{ autogenerate: { directory: '15-agent-rl' } }],
				},
				{
					label: '模块 16：多模态扩展',
					items: [{ autogenerate: { directory: '16-multimodal-extensions' } }],
				},
				{
					label: '附录：Capstone 项目',
					items: [{ autogenerate: { directory: 'appendix' } }],
				},
			],
		}),
	],
});
