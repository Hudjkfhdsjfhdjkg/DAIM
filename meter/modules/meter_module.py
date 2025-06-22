import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

from . import swin_transformer as swin
from . import objectives, meter_utils
from .bert import BertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def freeze_layers(model, bool):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = bool


class METERTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.all_text_feats = []
        self.all_image_feats = []
        self.save_hyperparameters()

        self.eval_bool = False

        resolution_after = config['image_size']

        self.cross_modal_text_transform = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform.apply(objectives.init_weights)
        self.cross_modal_image_transform = nn.Linear(config['input_image_embed_size'], config['hidden_size'])
        self.cross_modal_image_transform.apply(objectives.init_weights)

        #
        self.cross_modal_text_transform_1 = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform_1.apply(objectives.init_weights)
        self.cross_modal_text_transform_2 = nn.Linear(config['input_text_embed_size'], config['hidden_size'])
        self.cross_modal_text_transform_2.apply(objectives.init_weights)

        self.cross_modal_image_transform_1 = nn.Linear(256, config['hidden_size'])
        self.cross_modal_image_transform_1.apply(objectives.init_weights)
        self.cross_modal_image_transform_2 = nn.Linear(512, config['hidden_size'])
        self.cross_modal_image_transform_2.apply(objectives.init_weights)
        #

        self.vit_model = getattr(swin, self.hparams.config["vit"])(
            pretrained=True, config=self.hparams.config,
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.text_transformer = BertModel.from_pretrained(config['tokenizer'])

        freeze_layers(self.text_transformer.encoder, False)
        freeze_layers(self.text_transformer.embeddings, False)
        freeze_layers(self.text_transformer.pooler, False)
        freeze_layers(self.vit_model, False)

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
    ):
        print("infer batch type", type(batch))
        if img is None:
            print("yes")
            print("yes batch type", type(batch))
            if f"image_{image_token_type_idx - 1}" in batch:
                imgkey = f"image_{image_token_type_idx - 1}"
            else:
                print("no batch type", type(batch))
                print("no")
                imgkey = "image"
            img = batch[imgkey][0]




        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        # text_labels = batch[f"text_labels{do_mlm}"]
        text_types = batch[f"text_types"]
        text_masks = batch[f"text_masks"]

        text_lengths = batch[f'lengths']

        all_encoder_layers, pooled_output = self.text_transformer(text_ids, token_type_ids=text_types,
                                                                  attention_mask=text_masks)

        text_embeds = all_encoder_layers[-1]
        text_embeds = self.cross_modal_text_transform(text_embeds)

        #
        text_embeds_1 = all_encoder_layers[3]
        text_embeds_1 = self.cross_modal_text_transform_1(text_embeds_1)
        text_embeds_2 = all_encoder_layers[9]
        text_embeds_2 = self.cross_modal_text_transform_2(text_embeds_2)
        #

        image_embeds_all = self.vit_model(img)
        image_embeds = self.cross_modal_image_transform(image_embeds_all[-1])
        image_embeds_1 = self.cross_modal_image_transform_1(image_embeds_all[1])
        image_embeds_2 = self.cross_modal_image_transform_2(image_embeds_all[2])





        ret = {
            "text_feats": text_embeds,
            "image_feats": image_embeds,

            # "text_feats_0": text_embeds_0,
            "text_feats_1": text_embeds_1,
            "text_feats_2": text_embeds_2,
            # 'image_feats_0': image_embeds_0,
            'image_feats_1': image_embeds_1,
            'image_feats_2': image_embeds_2,
            # "cls_feats": cls_feats,
            # "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            'text_lengths': text_lengths,
        }

        return ret

    def forward(self, batch):
        print("forward batch type", type(batch))
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret




        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        # SNLI Visual Entailment
        if "snli" in self.current_tasks:
            ret.update(objectives.compute_snli(self, batch))

        # Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr_my(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        self.eval_bool = True
        print("training_step batch type", type(batch))


        meter_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])



        return total_loss

    def training_epoch_end(self, outs):
        pass





    def validation_step(self, batch, batch_idx):
        # 调用 infer 方法提取图文特征并保存

        images, input_ids, lengths, ids, input_mask, input_type_ids = batch
        ret = self.infer(
            {
                "image": [images],
                "text_ids": input_ids,
                "text_masks": input_mask,
                "text_types": input_type_ids,
                'lengths': lengths,
            }
        )
        # 保存或者使用返回的特征
        text_feats = ret["text_feats"]
        image_feats = ret["image_feats"]

        text_embeds = torch.mean(text_feats, dim=1)  # [58, 768]
        image_embeds = torch.mean(image_feats, dim=1)  # [58, 768]

        # 将特征添加到列表中以便后续可视化
        self.all_text_feats.append(text_embeds.cpu().numpy())
        self.all_image_feats.append(image_embeds.cpu().numpy())


    def validation_epoch_end(self, outs):
        # meter_utils.epoch_wrapup(self)
        if len(self.all_text_feats) == 0 or len(self.all_image_feats) == 0:
            print("没有收集到任何图文特征，跳过可视化")
            return

            # 输出 batch 数
        num_batches = len(outs)  # 或者 len(self.all_text_feats)

        # 合并所有批次的特征
        all_text_feats = np.concatenate(self.all_text_feats, axis=0)
        all_image_feats = np.concatenate(self.all_image_feats, axis=0)
        # 输出 batch 数
        num_batches = len(outs)  # 或者 len(self.all_text_feats)
        print(f"本次验证共 {num_batches} 个批次，共 {all_text_feats.shape[0]} 个图文对样本")

        # 可视化所有批次的图文特征
        self.visualize_features(all_text_feats, all_image_feats)

        # 清空列表，准备下一个 epoch
        self.all_text_feats = []
        self.all_image_feats = []

        if self.current_epoch != 0:
            meter_utils.epoch_eval_irtr(self)

        if self.current_epoch == 10:
            freeze_layers(self.vit_model, True)

            freeze_layers(self.text_transformer.encoder, True)
            freeze_layers(self.text_transformer.embeddings, True)
        # assert 1==0

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outs):
        # meter_utils.epoch_eval_irtr(self)
        meter_utils.epoch_eval_irtr(self, is_test=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
        return [optimizer], [scheduler]
        # return meter_utils.set_schedule(self)

    # 可视化函数（如前所示）
    def visualize_features(self,text_embeds, image_embeds):
        all_embeddings = np.concatenate([text_embeds, image_embeds], axis=0)

        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(all_embeddings)
        plt.rcParams['font.size'] = 14

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_embeddings[:len(text_embeds), 0], reduced_embeddings[:len(text_embeds), 1], label='Text',
                    alpha=0.6)
        plt.scatter(reduced_embeddings[len(text_embeds):, 0], reduced_embeddings[len(text_embeds):, 1], label='Image',
                    alpha=0.6)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14)

        plt.tight_layout()


        plt.show()
        plt.savefig("swin_bert_visualization.png")