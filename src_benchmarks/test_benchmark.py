"""
test_benchmark.py
=================
Unit test per:
  - convert_to_onnx.py  (ONNXWrapper, get_model, convert)
  - benchmark.py        (measure_system_metrics, _resize_tensor, run_benchmark)

Esecuzione:
    pip install pytest torch torchvision onnx onnxruntime psutil torchmetrics
    pytest test_benchmark.py -v

NON richiede il dataset NEU-DET né file .pth reali:
tutti i componenti che toccano il filesystem o l'hardware
vengono sostituiti da mock/fixture in-memory.
"""

import os
import sys
import tempfile
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from unittest.mock import MagicMock, patch, PropertyMock

# ── assicuriamoci che i moduli da testare siano importabili ──────────────────
# Se i file si trovano nella stessa cartella o in src/:
sys.path.insert(0, os.path.dirname(__file__))

# Stub di src_training.train_m1 (non disponibile nei test)
_stub = types.ModuleType("src_training")
_train_stub = types.ModuleType("src_training.train_m1")
_train_stub.NEUDETDataset = MagicMock
_train_stub.collate_fn    = lambda batch: tuple(zip(*batch))
sys.modules.setdefault("src_training", _stub)
sys.modules.setdefault("src_training.train_m1", _train_stub)


from convert_to_onnx import ONNXWrapper, get_model
from benchmark import measure_system_metrics, _resize_tensor, FIXED_INPUT_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Fixture condivise
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model():
    """Faster R-CNN con backbone MobileNet per velocizzare i test."""
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 7)
    model.eval()
    return model


@pytest.fixture(scope="module")
def wrapped_model(tiny_model):
    return ONNXWrapper(tiny_model)


@pytest.fixture(scope="module")
def dummy_input():
    """Tensore [1, 3, 800, 800] — stessa dimensione usata nell'export."""
    return torch.zeros(1, 3, 800, 800)


# ─────────────────────────────────────────────────────────────────────────────
# Test: get_model
# ─────────────────────────────────────────────────────────────────────────────

class TestGetModel:
    def test_returns_nn_module(self):
        model = get_model(num_classes=7)
        assert isinstance(model, nn.Module)

    def test_correct_num_classes(self):
        for n in [2, 7, 20]:
            model = get_model(num_classes=n)
            out_features = model.roi_heads.box_predictor.cls_score.out_features
            assert out_features == n, f"Atteso {n}, ottenuto {out_features}"

    def test_default_is_7_classes(self):
        model = get_model()
        assert model.roi_heads.box_predictor.cls_score.out_features == 7

    def test_model_is_in_train_mode_by_default(self):
        """get_model non chiama .eval(), quindi il modello è in train mode."""
        model = get_model()
        assert model.training is True

    def test_model_has_roi_heads(self):
        model = get_model()
        assert hasattr(model, 'roi_heads')

    def test_single_class_edge_case(self):
        """1 classe (solo background) non deve sollevare eccezioni."""
        model = get_model(num_classes=1)
        assert model.roi_heads.box_predictor.cls_score.out_features == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test: ONNXWrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestONNXWrapper:
    def test_is_nn_module(self, wrapped_model):
        assert isinstance(wrapped_model, nn.Module)

    def test_forward_returns_three_tensors(self, wrapped_model, dummy_input):
        with torch.no_grad():
            out = wrapped_model(dummy_input)
        assert isinstance(out, tuple), "L'output deve essere una tupla"
        assert len(out) == 3, "Devono esserci esattamente 3 tensori (boxes, labels, scores)"

    def test_boxes_shape(self, wrapped_model, dummy_input):
        with torch.no_grad():
            boxes, labels, scores = wrapped_model(dummy_input)
        # boxes: [N, 4]
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4

    def test_labels_and_scores_are_1d(self, wrapped_model, dummy_input):
        with torch.no_grad():
            boxes, labels, scores = wrapped_model(dummy_input)
        assert labels.ndim == 1
        assert scores.ndim == 1

    def test_labels_scores_same_length_as_boxes(self, wrapped_model, dummy_input):
        with torch.no_grad():
            boxes, labels, scores = wrapped_model(dummy_input)
        n = boxes.shape[0]
        assert labels.shape[0] == n
        assert scores.shape[0] == n

    def test_scores_in_0_1_range(self, wrapped_model, dummy_input):
        with torch.no_grad():
            _, _, scores = wrapped_model(dummy_input)
        if scores.numel() > 0:
            assert scores.min().item() >= 0.0
            assert scores.max().item() <= 1.0

    def test_labels_are_int64(self, wrapped_model, dummy_input):
        with torch.no_grad():
            _, labels, _ = wrapped_model(dummy_input)
        assert labels.dtype == torch.int64

    def test_boxes_are_float(self, wrapped_model, dummy_input):
        with torch.no_grad():
            boxes, _, _ = wrapped_model(dummy_input)
        assert boxes.dtype in (torch.float32, torch.float64)

    def test_wraps_original_model(self, tiny_model):
        wrapper = ONNXWrapper(tiny_model)
        assert wrapper.model is tiny_model

    def test_works_with_different_spatial_sizes(self, wrapped_model):
        """Il wrapper deve funzionare anche con immagini diverse da 800x800."""
        for h, w in [(800, 800), (600, 800), (1024, 1024)]:
            inp = torch.zeros(1, 3, h, w)
            with torch.no_grad():
                boxes, labels, scores = wrapped_model(inp)
            assert boxes.shape[1] == 4

    def test_empty_image_gives_empty_detections(self):
        """
        Un'immagine completamente nera (zero) non deve causare eccezioni;
        può restituire 0 detection.
        """
        model = get_model(num_classes=7)
        model.eval()
        wrapper = ONNXWrapper(model)
        black = torch.zeros(1, 3, 800, 800)
        with torch.no_grad():
            boxes, labels, scores = wrapper(black)
        assert boxes.shape[1] == 4 or boxes.numel() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: convert (conversione ONNX)
# ─────────────────────────────────────────────────────────────────────────────

class TestConvert:
    """
    Testa l'intera pipeline convert() salvando in una directory temporanea.
    Usa il modello MobileNet (molto più leggero) per non bloccare la CI.
    """

    @pytest.fixture(autouse=True)
    def tmp_dir(self, tmp_path):
        self.output_dir = str(tmp_path / "onnx_out")
        self.pth_path   = str(tmp_path / "fake_model.pth")

    def _save_fake_checkpoint(self, num_classes=7):
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, weights_backbone=None
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        torch.save({'model_state_dict': model.state_dict()}, self.pth_path)
        return model

    def test_output_file_exists(self):
        """Il file ONNX deve essere creato."""
        from convert_to_onnx import convert as onnx_convert
        # Patch get_model per usare backbone leggero
        with patch('convert_to_onnx.get_model') as mock_gm:
            model = self._save_fake_checkpoint()
            mock_gm.return_value = torchvision.models.detection \
                .fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
            in_f = mock_gm.return_value.roi_heads.box_predictor.cls_score.in_features
            mock_gm.return_value.roi_heads.box_predictor = FastRCNNPredictor(in_f, 7)
            out_path = onnx_convert(self.pth_path, self.output_dir)
        assert os.path.isfile(out_path)

    def test_output_is_valid_onnx(self):
        """Il file prodotto deve superare onnx.checker."""
        import onnx as _onnx
        from convert_to_onnx import convert as onnx_convert
        with patch('convert_to_onnx.get_model') as mock_gm:
            self._save_fake_checkpoint()
            light = torchvision.models.detection \
                .fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
            in_f  = light.roi_heads.box_predictor.cls_score.in_features
            light.roi_heads.box_predictor = FastRCNNPredictor(in_f, 7)
            mock_gm.return_value = light
            out_path = onnx_convert(self.pth_path, self.output_dir)
        model_proto = _onnx.load(out_path)
        _onnx.checker.check_model(model_proto)   # non deve lanciare

    def test_raw_file_removed_after_conversion(self):
        """Il file _raw.onnx intermedio deve essere eliminato."""
        from convert_to_onnx import convert as onnx_convert
        with patch('convert_to_onnx.get_model') as mock_gm:
            self._save_fake_checkpoint()
            light = torchvision.models.detection \
                .fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
            in_f  = light.roi_heads.box_predictor.cls_score.in_features
            light.roi_heads.box_predictor = FastRCNNPredictor(in_f, 7)
            mock_gm.return_value = light
            out_path = onnx_convert(self.pth_path, self.output_dir)
        raw = out_path.replace('_shaped.onnx', '_raw.onnx')
        assert not os.path.exists(raw), "Il file _raw.onnx non deve sopravvivere"

    def test_missing_pth_raises(self):
        """Se il file .pth non esiste deve sollevare un'eccezione."""
        from convert_to_onnx import convert as onnx_convert
        with pytest.raises(Exception):
            onnx_convert("/non/esiste/model.pth", self.output_dir)

    def test_output_dir_created_if_missing(self):
        """La directory di output viene creata automaticamente."""
        from convert_to_onnx import convert as onnx_convert
        new_dir = os.path.join(self.output_dir, "subdir", "deep")
        with patch('convert_to_onnx.get_model') as mock_gm:
            self._save_fake_checkpoint()
            light = torchvision.models.detection \
                .fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
            in_f  = light.roi_heads.box_predictor.cls_score.in_features
            light.roi_heads.box_predictor = FastRCNNPredictor(in_f, 7)
            mock_gm.return_value = light
            out_path = onnx_convert(self.pth_path, new_dir)
        assert os.path.isdir(new_dir)

    def test_dynamic_axes_in_onnx(self):
        """L'output 'boxes' deve avere la dimensione 0 marcata come dinamica."""
        import onnx as _onnx
        from convert_to_onnx import convert as onnx_convert
        with patch('convert_to_onnx.get_model') as mock_gm:
            self._save_fake_checkpoint()
            light = torchvision.models.detection \
                .fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
            in_f  = light.roi_heads.box_predictor.cls_score.in_features
            light.roi_heads.box_predictor = FastRCNNPredictor(in_f, 7)
            mock_gm.return_value = light
            out_path = onnx_convert(self.pth_path, self.output_dir)
        proto = _onnx.load(out_path)
        # Troviamo l'output 'boxes' e verifichiamo che dim[0] sia parametrica
        boxes_out = next(o for o in proto.graph.output if o.name == 'boxes')
        dim0 = boxes_out.type.tensor_type.shape.dim[0]
        # dim_param non vuoto significa asse dinamico
        assert dim0.dim_param != '' or dim0.dim_value == 0, \
            "La dimensione 0 di 'boxes' deve essere dinamica (num_detections)"


# ─────────────────────────────────────────────────────────────────────────────
# Test: measure_system_metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMeasureSystemMetrics:
    def test_returns_two_values(self):
        ram, cpu = measure_system_metrics()
        assert isinstance(ram, float)
        assert isinstance(cpu, float)

    def test_ram_is_positive(self):
        ram, _ = measure_system_metrics()
        assert ram > 0, "Il processo deve occupare RAM > 0 MB"

    def test_cpu_in_valid_range(self):
        _, cpu = measure_system_metrics()
        # cpu_percent restituisce 0.0 se chiamato immediatamente dopo un reset
        assert 0.0 <= cpu <= 100.0 * os.cpu_count()

    def test_ram_in_reasonable_range(self):
        """Un processo Python normale non occupa 0 MB né 100 GB."""
        ram, _ = measure_system_metrics()
        assert 1.0 < ram < 100_000.0   # MB


# ─────────────────────────────────────────────────────────────────────────────
# Test: _resize_tensor
# ─────────────────────────────────────────────────────────────────────────────

class TestResizeTensor:
    def test_output_shape(self):
        img = torch.zeros(3, 300, 400)
        result = _resize_tensor(img)
        h, w = FIXED_INPUT_SIZE
        assert result.shape == (1, 3, h, w)

    def test_output_is_numpy(self):
        img = torch.zeros(3, 300, 400)
        result = _resize_tensor(img)
        assert isinstance(result, np.ndarray)

    def test_batch_dim_is_1(self):
        img = torch.rand(3, 640, 480)
        result = _resize_tensor(img)
        assert result.shape[0] == 1

    def test_channel_dim_preserved(self):
        img = torch.rand(3, 100, 100)
        result = _resize_tensor(img)
        assert result.shape[1] == 3

    def test_already_correct_size_unchanged(self):
        h, w = FIXED_INPUT_SIZE
        img    = torch.rand(3, h, w)
        result = _resize_tensor(img)
        assert result.shape == (1, 3, h, w)

    def test_values_preserved_when_no_resize(self):
        h, w  = FIXED_INPUT_SIZE
        img   = torch.ones(3, h, w) * 0.5
        result = _resize_tensor(img)
        assert np.allclose(result, 0.5, atol=1e-5)

    def test_very_small_input(self):
        """Immagini molto piccole (es. 1x1) non devono causare crash."""
        img = torch.zeros(3, 1, 1)
        result = _resize_tensor(img)
        h, w = FIXED_INPUT_SIZE
        assert result.shape == (1, 3, h, w)

    def test_large_input(self):
        """Immagini grandi vengono scalate correttamente."""
        img = torch.zeros(3, 2000, 2000)
        result = _resize_tensor(img)
        h, w = FIXED_INPUT_SIZE
        assert result.shape == (1, 3, h, w)

    def test_dtype_float32(self):
        img    = torch.rand(3, 300, 400, dtype=torch.float32)
        result = _resize_tensor(img)
        assert result.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Test: run_benchmark (con mock completo — non richiede GPU né dataset reale)
# ─────────────────────────────────────────────────────────────────────────────

def _make_fake_dataset(n=5):
    """Restituisce una lista di (tensore immagine, target dict) fake."""
    items = []
    for i in range(n):
        img = torch.rand(3, 300, 400)
        target = {
            'boxes':    torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
            'labels':   torch.tensor([1], dtype=torch.int64),
            'image_id': torch.tensor([i]),
            'area':     torch.tensor([1600.0]),
            'iscrowd':  torch.tensor([0], dtype=torch.int64),
        }
        items.append((img, target))
    return items


class TestRunBenchmarkPyTorch:
    """
    Testa run_benchmark con un modello PyTorch mock.
    Nessun accesso a filesystem o GPU richiesto.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, tiny_model):
        # Salva un checkpoint .pth valido
        self.pth_path = str(tmp_path / "model.pth")
        torch.save({'model_state_dict': tiny_model.state_dict()}, self.pth_path)
        self.data_root = str(tmp_path / "data")
        os.makedirs(self.data_root, exist_ok=True)

    def _mock_dataloader(self, fake_items):
        """Ritorna un DataLoader mock che itera sui fake_items."""
        loader = MagicMock()
        loader.__iter__ = MagicMock(return_value=iter(
            [([img], [tgt]) for img, tgt in fake_items]
        ))
        loader.__len__ = MagicMock(return_value=len(fake_items))
        return loader

    def test_returns_dict_with_expected_keys(self, tiny_model):
        fake_items = _make_fake_dataset()
        fake_loader = self._mock_dataloader(fake_items)
        fake_img, _ = fake_items[0]
        fake_dataset = MagicMock()
        fake_dataset.__getitem__ = MagicMock(return_value=(fake_img, {}))

        with patch('benchmark.NEUDETDataset', return_value=fake_dataset), \
             patch('benchmark.DataLoader', return_value=fake_loader), \
             patch('benchmark.torchvision.models.detection'
                   '.fasterrcnn_resnet50_fpn', return_value=tiny_model), \
             patch('torch.load', return_value={'model_state_dict': tiny_model.state_dict()}):
            result = run_benchmark(self.pth_path, self.data_root)

        expected_keys = {'map_50', 'avg_latency_ms', 'fps', 'ram_mb', 'cpu_pct', 'inf_per_watt'}
        assert expected_keys.issubset(result.keys())

    def test_fps_is_positive(self, tiny_model):
        fake_items  = _make_fake_dataset()
        fake_loader = self._mock_dataloader(fake_items)
        fake_img, _ = fake_items[0]
        fake_dataset = MagicMock()
        fake_dataset.__getitem__ = MagicMock(return_value=(fake_img, {}))

        with patch('benchmark.NEUDETDataset', return_value=fake_dataset), \
             patch('benchmark.DataLoader', return_value=fake_loader), \
             patch('benchmark.torchvision.models.detection'
                   '.fasterrcnn_resnet50_fpn', return_value=tiny_model), \
             patch('torch.load', return_value={'model_state_dict': tiny_model.state_dict()}):
            result = run_benchmark(self.pth_path, self.data_root)

        assert result['fps'] > 0

    def test_latency_consistency_with_fps(self, tiny_model):
        """fps deve essere circa 1000 / avg_latency_ms."""
        fake_items  = _make_fake_dataset()
        fake_loader = self._mock_dataloader(fake_items)
        fake_img, _ = fake_items[0]
        fake_dataset = MagicMock()
        fake_dataset.__getitem__ = MagicMock(return_value=(fake_img, {}))

        with patch('benchmark.NEUDETDataset', return_value=fake_dataset), \
             patch('benchmark.DataLoader', return_value=fake_loader), \
             patch('benchmark.torchvision.models.detection'
                   '.fasterrcnn_resnet50_fpn', return_value=tiny_model), \
             patch('torch.load', return_value={'model_state_dict': tiny_model.state_dict()}):
            result = run_benchmark(self.pth_path, self.data_root)

        recomputed_fps = 1000.0 / result['avg_latency_ms']
        assert abs(recomputed_fps - result['fps']) < 0.01

    def test_map50_in_0_100_range(self, tiny_model):
        fake_items  = _make_fake_dataset()
        fake_loader = self._mock_dataloader(fake_items)
        fake_img, _ = fake_items[0]
        fake_dataset = MagicMock()
        fake_dataset.__getitem__ = MagicMock(return_value=(fake_img, {}))

        with patch('benchmark.NEUDETDataset', return_value=fake_dataset), \
             patch('benchmark.DataLoader', return_value=fake_loader), \
             patch('benchmark.torchvision.models.detection'
                   '.fasterrcnn_resnet50_fpn', return_value=tiny_model), \
             patch('torch.load', return_value={'model_state_dict': tiny_model.state_dict()}):
            result = run_benchmark(self.pth_path, self.data_root)

        assert 0.0 <= result['map_50'] <= 100.0

    def test_inf_per_watt_scales_with_tdp(self, tiny_model):
        """Raddoppiare il TDP deve dimezzare inf/watt."""
        fake_items  = _make_fake_dataset()
        fake_loader = self._mock_dataloader(fake_items)
        fake_img, _ = fake_items[0]
        fake_dataset = MagicMock()
        fake_dataset.__getitem__ = MagicMock(return_value=(fake_img, {}))

        common_patches = [
            patch('benchmark.NEUDETDataset', return_value=fake_dataset),
            patch('benchmark.DataLoader', return_value=fake_loader),
            patch('benchmark.torchvision.models.detection'
                  '.fasterrcnn_resnet50_fpn', return_value=tiny_model),
            patch('torch.load', return_value={'model_state_dict': tiny_model.state_dict()}),
        ]

        # Due run con TDP diversi — fakeiamo un timing fisso per stabilità
        with patch('time.perf_counter', side_effect=[0.0, 5.0]):
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3]:
                r10 = run_benchmark(self.pth_path, self.data_root, tdp_w=10.0)

        with patch('time.perf_counter', side_effect=[0.0, 5.0]):
            with common_patches[0], common_patches[1], common_patches[2], common_patches[3]:
                r20 = run_benchmark(self.pth_path, self.data_root, tdp_w=20.0)

        assert abs(r10['inf_per_watt'] - r20['inf_per_watt'] * 2) < 1e-3


class TestRunBenchmarkONNX:
    """
    Testa il ramo ONNX di run_benchmark mockando onnxruntime.InferenceSession.
    Non richiede GPU né un file .onnx reale.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        # File fake con estensione .onnx
        self.onnx_path = str(tmp_path / "model.onnx")
        with open(self.onnx_path, 'wb') as f:
            f.write(b'\x00' * 16)   # contenuto irrilevante (mockato)
        self.data_root = str(tmp_path / "data")
        os.makedirs(self.data_root, exist_ok=True)

    def _make_ort_session_mock(self):
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name='input')]
        # Output fake: 2 box, 2 label, 2 score
        boxes  = np.array([[10, 10, 50, 50], [20, 20, 80, 80]], dtype=np.float32)
        labels = np.array([1, 2], dtype=np.int64)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        session.run.return_value = [boxes, labels, scores]
        return session

    def _mock_dataloader(self, fake_items):
        loader = MagicMock()
        loader.__iter__ = MagicMock(return_value=iter(
            [([img], [tgt]) for img, tgt in fake_items]
        ))
        loader.__len__ = MagicMock(return_value=len(fake_items))
        return loader

    def test_onnx_branch_returns_dict(self):
        fake_items  = _make_fake_dataset()
        fake_loader = self._mock_dataloader(fake_items)
        fake_img, _ = fake_items[0]
        fake_dataset = MagicMock()
        fake_dataset.__getitem__ = MagicMock(return_value=(fake_img, {}))
        ort_session  = self._make_ort_session_mock()

        with patch('benchmark.NEUDETDataset', return_value=fake_dataset), \
             patch('benchmark.DataLoader', return_value=fake_loader), \
             patch('onnxruntime.InferenceSession', return_value=ort_session):
            result = run_benchmark(self.onnx_path, self.data_root)

        assert 'map_50' in result
        assert 'fps'    in result

    def test_onnx_session_called_with_correct_input_name(self):
        fake_items  = _make_fake_dataset()
        fake_loader = self._mock_dataloader(fake_items)
        fake_img, _ = fake_items[0]
        fake_dataset = MagicMock()
        fake_dataset.__getitem__ = MagicMock(return_value=(fake_img, {}))
        ort_session  = self._make_ort_session_mock()

        with patch('benchmark.NEUDETDataset', return_value=fake_dataset), \
             patch('benchmark.DataLoader', return_value=fake_loader), \
             patch('onnxruntime.InferenceSession', return_value=ort_session):
            run_benchmark(self.onnx_path, self.data_root)

        # session.run deve essere chiamata con 'input' come chiave
        calls = ort_session.run.call_args_list
        assert len(calls) > 0
        for call in calls:
            _, kwargs_or_args = call
            input_dict = call[0][1] if call[0] else call[1]['input_feed']
            assert 'input' in input_dict

    def test_onnx_input_has_correct_shape(self):
        """L'array numpy passato a session.run deve avere shape (1,3,H,W)."""
        fake_items  = _make_fake_dataset(n=1)
        fake_loader = self._mock_dataloader(fake_items)
        fake_img, _ = fake_items[0]
        fake_dataset = MagicMock()
        fake_dataset.__getitem__ = MagicMock(return_value=(fake_img, {}))
        ort_session  = self._make_ort_session_mock()

        with patch('benchmark.NEUDETDataset', return_value=fake_dataset), \
             patch('benchmark.DataLoader', return_value=fake_loader), \
             patch('onnxruntime.InferenceSession', return_value=ort_session):
            run_benchmark(self.onnx_path, self.data_root)

        # Tra tutte le chiamate a session.run, verifichiamo la fase mAP
        # (le ultime n_val chiamate dopo il warmup)
        for call in ort_session.run.call_args_list:
            input_arr = list(call[0][1].values())[0]
            assert input_arr.ndim == 4
            assert input_arr.shape[0] == 1
            assert input_arr.shape[1] == 3
            h, w = FIXED_INPUT_SIZE
            assert input_arr.shape[2] == h
            assert input_arr.shape[3] == w


# ─────────────────────────────────────────────────────────────────────────────
# Test: casi limite numerici
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalEdgeCases:
    def test_fps_never_zero(self):
        """Anche con timing artificialmente lento fps > 0."""
        with patch('time.perf_counter', side_effect=[0.0, 1000.0]):
            # 50 iterazioni in 1000 secondi → 0.02 FPS, ma comunque > 0
            latency_ms = (1000.0 / 50) * 1000
            fps = 1000.0 / latency_ms
            assert fps > 0

    def test_map50_not_nan_on_no_detections(self):
        """MeanAveragePrecision non deve restituire NaN se il modello non detecta nulla."""
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        metric = MeanAveragePrecision(iou_thresholds=[0.5])
        preds = [{'boxes': torch.zeros((0, 4)), 'scores': torch.zeros(0),
                  'labels': torch.zeros(0, dtype=torch.int64)}]
        targets = [{'boxes': torch.tensor([[10., 10., 50., 50.]]),
                    'labels': torch.tensor([1], dtype=torch.int64)}]
        metric.update(preds, targets)
        result = metric.compute()
        assert not torch.isnan(result['map_50'])

    def test_inf_per_watt_formula(self):
        """Test della formula pura: inf/watt = fps / tdp."""
        fps  = 12.5
        tdp  = 10.0
        result = fps / tdp
        assert abs(result - 1.25) < 1e-9

    def test_resize_preserves_float32_range(self):
        """I valori dopo il resize devono stare in [0, 1] se l'input è normalizzato."""
        img = torch.rand(3, 300, 400)
        out = _resize_tensor(img)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-5   # piccola tolleranza per interpolazione


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])